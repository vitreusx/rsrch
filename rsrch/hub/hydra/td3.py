import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler2 import Profiler2
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from . import env
from .utils import TruncNormal2


@dataclass
class Config:
    seed: int = 1
    device: str = "cuda"
    env_id: str = "Ant-v4"
    total_steps: int = int(1e6)
    train_envs: int = 1
    buffer_cap: int = int(100e3)
    warmup: int = int(25e3)
    env_batch: int = 1
    batch_size: int = 256
    opt_iters: int = 1
    policy_opt_freq: float = 0.5
    expl_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    gamma: float = 0.99
    tau: float = 0.995
    val_episodes: int = 16
    val_envs: int = val_episodes
    val_every: int = int(32e3)


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
    train_envs = env_f.vector_env(cfg.train_envs)
    val_envs = env_f.vector_env(cfg.val_envs)

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
            # self.enc.apply(layer_init)
            self.head = nn.Sequential(
                nn.Linear(256, 1),
                nn.Flatten(0),
            )
            # self.head.apply(partial(layer_init, std=1.0))
            self.to(device)

        def forward(self, obs: Tensor, act: Tensor):
            obs, act = obs.flatten(1), act.flatten(1)
            net_x = torch.cat([obs, act], -1)
            return self.head(self.enc(net_x))

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.register_buffer(
                "_loc", 0.5 * (env_f.act_space.low + env_f.act_space.high)
            )
            self.register_buffer(
                "_scale", 0.5 * (env_f.act_space.high - env_f.act_space.low)
            )
            # self.enc.apply(layer_init)
            self.head = nn.Linear(256, act_dim)
            # self.head.apply(layer_init)
            self.to(device)

        def forward(self, obs: Tensor) -> Tensor:
            act = self.head(self.enc(obs))
            act = act.reshape(-1, *env_f.act_space.shape)
            act = self._loc + self._scale * act
            act = act.clamp(env_f.act_space.low, env_f.act_space.high)
            return act

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

    sampler = data.UniformSampler()
    buf = env_f.step_buffer(cfg.buffer_cap, sampler)
    ep_ids = [None for _ in range(train_envs.num_envs)]
    ep_rets = [0.0 for _ in range(train_envs.num_envs)]
    ep_lens = [0 for _ in range(train_envs.num_envs)]

    env_step = 0
    exp = tensorboard.Experiment("hydra", prefix=cfg.env_id)
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(desc="Hydra", total=cfg.total_steps)
    prof = Profiler2(exp.dir / "traces", device)

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

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs)
            return env_f.move_act(act, to="env")

    train_agent, val_agent = TrainAgent(), ValAgent()
    env_iter = iter(rollout.steps(train_envs, train_agent))

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_log_q = cron.Every(lambda: env_step, 128)
    should_log_a = cron.Every(lambda: env_step, 128)
    q_opt_iters, actor_opt_iters = 0, 0

    with prof(name="Hydra") as _prof:
        while env_step < cfg.total_steps:
            if should_val:
                actor.eval()
                val_eps = iter(
                    rollout.episodes(val_envs, val_agent, max_episodes=cfg.val_episodes)
                )
                val_rets = []
                for _, ep in val_eps:
                    val_rets.append(sum(ep.reward))
                exp.add_scalar("val/mean_ret", np.mean(val_rets))
                actor.train()

            for _ in range(cfg.env_batch):
                env_idx, step = next(env_iter)
                ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)

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
                    _prof.step()

            if env_step < cfg.warmup:
                continue

            for _ in range(cfg.opt_iters):
                idxes = sampler.sample(cfg.batch_size)
                batch = env_f.fetch_step_batch(buf, idxes)

                with torch.no_grad():
                    noise = torch.randn_like(batch.act) * cfg.policy_noise
                    noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
                    noise = noise * act_scale
                    next_act = actor_t(batch.next_obs) + noise
                    next_act = next_act.clamp(env_f.act_space.low, env_f.act_space.high)

                    next_q1_t = q1_t(batch.next_obs, next_act)
                    next_q2_t = q2_t(batch.next_obs, next_act)
                    q_target = batch.reward + cfg.gamma * (
                        1.0 - batch.term.float()
                    ) * torch.min(next_q1_t, next_q2_t)

                q1_pred = q1(batch.obs, batch.act)
                q1_loss = F.mse_loss(q1_pred, q_target)
                q2_pred = q2(batch.obs, batch.act)
                q2_loss = F.mse_loss(q2_pred, q_target)
                q_loss = q1_loss + q2_loss

                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                q_opt.step()
                q_opt_iters += 1

                if should_log_q:
                    exp.add_scalar("train/q1_pred", q1_pred.mean())
                    exp.add_scalar("train/q2_pred", q2_pred.mean())
                    exp.add_scalar("train/q1_loss", q1_loss)
                    exp.add_scalar("train/q2_loss", q2_loss)
                    exp.add_scalar("train/q_loss", q_loss)
                    exp.add_scalar("train/q_opt_iters", q_opt_iters)

                if q_opt_iters * cfg.policy_opt_freq >= actor_opt_iters:
                    actor_loss = -q1(batch.obs, actor(batch.obs)).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()
                    actor_p.step()
                    q1_p.step()
                    q2_p.step()
                    actor_opt_iters += 1

                    if should_log_a:
                        exp.add_scalar("train/actor_loss", actor_loss)
                        exp.add_scalar("train/actor_opt_iters", actor_opt_iters)
