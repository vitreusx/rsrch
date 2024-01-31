import random
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import ImageSequenceClip
from PIL import Image
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler import profiler
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from . import env


@dataclass
class Config:
    seed: int = 1
    device: str = "cuda"
    env_id: str = "Ant-v4"
    total_steps: int = int(1e6)
    train_envs: int = 1
    buffer_cap: int = int(128e3)
    warmup: int = int(16e3)
    env_batch: int = 1
    batch_size: int = 32
    num_steps: int = 16
    opt_iters: int = 1
    policy_opt_freq: float = 0.5
    expl_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    gamma: float = 0.99
    tau: float = 0.995
    gae_lambda: float = 0.95
    val_episodes: int = 16
    val_envs: int = val_episodes
    val_every: int = int(32e3)
    rec_every: int = val_every
    prioritized: bool = False
    n_step: int = 1


def layer_init(layer, std=nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
    return layer


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)

    env_cfg = env.Config(type="gym", gym=env.gym.Config(env_id=cfg.env_id))
    env_f = env.make_factory(env_cfg, device)
    assert isinstance(env_f.obs_space, spaces.torch.Box)
    assert isinstance(env_f.act_space, spaces.torch.Box)

    train_envs = env_f.vector_env(cfg.train_envs, mode="train")
    val_envs = env_f.vector_env(cfg.val_envs, mode="val")
    rec_env = env_f.env(mode="val", record=True)

    obs_dim = int(np.prod(env_f.obs_space.shape))
    act_dim = int(np.prod(env_f.act_space.shape))

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.enc.apply(layer_init)
            self.head = nn.Sequential(
                nn.Linear(256, 1),
                nn.Flatten(0),
            )
            self.head.apply(partial(layer_init, std=1.0))
            self.to(device)

        def forward(self, obs: Tensor, act: Tensor):
            obs, act = obs.flatten(1), act.flatten(1)
            net_x = torch.cat([obs, act], -1)
            return self.head(self.enc(net_x))

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, act_dim),
            )
            self.register_buffer(
                "act_loc", 0.5 * (env_f.act_space.low + env_f.act_space.high)
            )
            self.register_buffer(
                "act_scale", 0.5 * (env_f.act_space.high - env_f.act_space.low)
            )
            self.to(device)

        def forward(self, obs: Tensor) -> Tensor:
            act = self.net(obs)
            act = act.reshape(-1, *env_f.act_space.shape)
            act = self.act_loc + self.act_scale * F.tanh(act)
            return act

    class WorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim + act_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, obs_dim),
            )
            layer_init(self.fc, std=1e-2)
            self.to(device)

        def forward(self, obs: Tensor, act: Tensor) -> Tensor:
            x = torch.cat([obs, act], -1)
            return obs + self.fc(x)

    wm = WorldModel()
    wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4, eps=1e-5)

    q1, q2, q1_t, q2_t = Q(), Q(), Q(), Q()
    q1_p = polyak.Polyak(q1, q1_t, cfg.tau)
    q2_p = polyak.Polyak(q2, q2_t, cfg.tau)
    q_params = [*q1.parameters(), *q2.parameters()]
    q_opt = torch.optim.Adam(q_params, lr=3e-4, eps=1e-5)

    actor, actor_t = Actor(), Actor()
    actor_p = polyak.Polyak(actor, actor_t, cfg.tau)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)

    polyak.sync(q1, q1_t)
    polyak.sync(q2, q2_t)
    polyak.sync(actor, actor_t)

    if cfg.prioritized:
        sampler = data.PrioritizedSampler(cfg.buffer_cap)
        max_loss = None
    else:
        sampler = data.UniformSampler()

    if cfg.n_step == 1:
        buf = env_f.step_buffer(cfg.buffer_cap, sampler)
    else:
        buf = env_f.slice_buffer(
            cfg.buffer_cap, num_steps=cfg.num_steps, sampler=sampler
        )

    ep_ids = [None for _ in range(train_envs.num_envs)]
    ep_rets = [0.0 for _ in range(train_envs.num_envs)]
    ep_lens = [0 for _ in range(train_envs.num_envs)]

    env_step = 0

    exp = tensorboard.Experiment(
        project="td3",
        run=f"{cfg.env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(desc="TD3", total=cfg.total_steps)
    vid_dir = exp.dir / "videos"

    prof = profiler(exp.dir / "traces", device)
    prof.__enter__()

    act_scale = 0.5 * (env_f.act_space.high - env_f.act_space.low)

    class TrainAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            if env_step < cfg.warmup:
                return env_f.env_act_space.sample([len(obs)])
            else:
                obs = env_f.move_obs(obs)
                act = actor(obs)
                noise = torch.randn_like(act) * cfg.expl_noise
                noise = noise * act_scale
                act = (act + noise).clamp(env_f.act_space.low, env_f.act_space.high)
                return env_f.move_act(act, to="env")

    train_agent = TrainAgent()
    env_iter = iter(rollout.steps(train_envs, train_agent))

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs)
            return env_f.move_act(act, to="env")

    val_agent = ValAgent()

    make_val_iter = lambda: rollout.episodes(
        val_envs,
        val_agent,
        max_episodes=cfg.val_episodes,
    )

    class RecAgent(gym.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs[None])
            act = actor(obs)
            return env_f.move_act(act, to="env")[0]

    rec_agent = RecAgent()

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    # should_rec = cron.Every(lambda: env_step, cfg.rec_every)
    should_rec = cron.Never()
    should_log_q = cron.Every(lambda: env_step, 128)
    should_log_a = cron.Every(lambda: env_step, 128)
    q_opt_iters, actor_opt_iters = 0, 0

    # gen_adv_est = GenAdvEst(gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)

    while env_step < cfg.total_steps:
        if should_val:
            actor.eval()

            val_iter = tqdm(
                make_val_iter(),
                total=cfg.val_episodes,
                leave=False,
                desc="Val",
            )

            val_rets = []
            for _, ep in val_iter:
                val_rets.append(sum(ep.reward))

            exp.add_scalar("val/mean_ret", np.mean(val_rets))
            actor.train()

        if should_rec:
            actor.eval()

            with TemporaryDirectory() as tmpdir:
                frame_idx, ep_ret = 0, 0.0
                obs, info = rec_env.reset()
                rec_agent.reset(obs, info)

                dest = Path(tmpdir) / f"{frame_idx:08d}.png"
                Image.fromarray(rec_env.render()).save(dest)

                while True:
                    act = rec_agent.policy(obs)
                    next_obs, rew, term, trunc, info = rec_env.step(act)
                    rec_agent.step(act)
                    rec_agent.observe(act, next_obs, rew, term, trunc, info)
                    ep_ret += rew
                    obs = next_obs

                    frame_idx += 1
                    dest = Path(tmpdir) / f"{frame_idx:08d}.png"
                    Image.fromarray(rec_env.render()).save(dest)

                    if term or trunc:
                        break

                vid = ImageSequenceClip(
                    tmpdir,
                    fps=rec_env.metadata.get("render_fps", 30),
                )

                dst = vid_dir / f"step={env_step}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                vid.write_videofile(str(dst), verbose=False, logger=None)

            actor.train()

        for _ in range(cfg.env_batch):
            env_idx, step = next(env_iter)
            ep_ids[env_idx], step_id = buf.push(ep_ids[env_idx], step)
            if isinstance(sampler, data.PrioritizedSampler):
                if max_loss is not None and step_id is not None:
                    sampler.update([step_id], [max_loss])

            ep_rets[env_idx] += step.reward
            ep_lens[env_idx] += 1

            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                ep_rets[env_idx] = 0.0
                exp.add_scalar("train/ep_len", ep_lens[env_idx])
                ep_lens[env_idx] = 0

            env_step += 1
            pbar.update(1)
            if env_step >= cfg.warmup:
                prof.step()

        if env_step < cfg.warmup:
            continue

        for _ in range(cfg.opt_iters):
            idxes, _ = sampler.sample(cfg.batch_size)

            # batch = env_f.fetch_slice_batch(buf, idxes)
            batch = env_f.fetch_step_batch(buf, idxes)

            with torch.no_grad():
                next_act = actor_t(batch.next_obs)
                noise = torch.randn_like(next_act) * cfg.policy_noise
                noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
                noise = noise * act_scale
                next_act = next_act + noise
                next_act = next_act.clamp(env_f.act_space.low, env_f.act_space.high)

                next_q1_t = q1_t(batch.next_obs, next_act)
                next_q2_t = q2_t(batch.next_obs, next_act)
                next_q = torch.min(next_q1_t, next_q2_t)
                q_target = (
                    batch.reward + cfg.gamma * (1.0 - batch.term.float()) * next_q
                )

            q1_pred = q1(batch.obs, batch.act)
            q1_loss = (q1_pred - q_target).square()
            q2_pred = q2(batch.obs, batch.act)
            q2_loss = (q2_pred - q_target).square()
            q_loss = q1_loss + q2_loss

            q_opt.zero_grad(set_to_none=True)
            q_loss.mean().backward()
            q_opt.step()
            q_opt_iters += 1

            if isinstance(sampler, data.PrioritizedSampler):
                q_loss_ = q_loss.detach().cpu().numpy()
                if max_loss is None:
                    max_loss = np.amax(q_loss_)
                    prio = max_loss * np.ones([len(buf)])
                    sampler.update([*buf.keys()], prio)

                sampler.update(idxes, q_loss_)
                max_loss = max(max_loss, np.amax(q_loss_))

            wm_pred = wm(batch.obs, batch.act)
            wm_pred_q = q1(wm_pred, next_act)
            wm_loss = F.mse_loss(wm_pred_q, next_q)

            wm_opt.zero_grad(set_to_none=True)
            wm_loss.backward()

            if should_log_q:
                exp.add_scalar("train/q1_pred", q1_pred.mean())
                exp.add_scalar("train/q2_pred", q2_pred.mean())
                exp.add_scalar("train/q1_loss", q1_loss.mean())
                exp.add_scalar("train/q2_loss", q2_loss.mean())
                exp.add_scalar("train/q_loss", q_loss.mean())
                exp.add_scalar("train/wm_loss", wm_loss)

            if q_opt_iters * cfg.policy_opt_freq >= actor_opt_iters:
                actor_loss = -q1(batch.obs, actor(batch.obs)).mean()
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()
                actor_opt_iters += 1

                actor_p.step()
                q1_p.step()
                q2_p.step()

                if should_log_a:
                    exp.add_scalar("train/actor_loss", actor_loss)
