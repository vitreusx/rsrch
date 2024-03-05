from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.nn import dist_head as dh
from rsrch.nn import fc
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro

from .. import env
from ..alpha import Alpha
from . import config


def main():
    cfg_d = config.cli(
        config_file=Path(__file__).parent / "config.yml",
    )

    cfg = config.from_dicts([cfg_d], config.Config)

    rs = repro.RandomState()
    rs.init(cfg.random.seed, cfg.random.deterministic)

    device = torch.device(cfg.device)

    env_f = env.make_factory(cfg.env, device)
    assert isinstance(env_f.obs_space, spaces.torch.Box)
    assert isinstance(env_f.act_space, spaces.torch.Box)

    obs_dim = int(np.prod(env_f.obs_space.shape))
    act_dim = int(np.prod(env_f.act_space.shape))

    exp = tensorboard.Experiment(project="sac", config=cfg_d)

    env_step, opt_step = 0, 0
    exp.register_step("env_step", lambda: env_step, default=True)
    exp.register_step("opt_step", lambda: opt_step)

    def make_sched(cfg):
        nonlocal env_step, opt_step
        count, unit = cfg["every"]
        step_fn = exp.get_step_fn(unit)
        cfg = {**cfg, "step_fn": step_fn, "every": count}
        return cron.Every2(**cfg)

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self._fc = nn.Sequential(
                fc.FullyConnected(
                    layer_sizes=[obs_dim + act_dim, cfg.hidden_dim, cfg.hidden_dim, 1],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                nn.Flatten(0),
            )

        def forward(self, s: Tensor, a: Tensor):
            x = torch.cat([s.flatten(1), a.flatten(1)], -1)
            return self._fc(x)

    qf, qf_t = nn.ModuleList(), nn.ModuleList()
    for _ in range(2):
        qf.append(Q().to(device))
        qf_t.append(Q().to(device))

    qf_opt = cfg.opt.make()(qf.parameters())
    polyak.sync(qf, qf_t)
    should_step_polyak = make_sched(cfg.polyak.sched)

    class Actor(nn.Sequential):
        def __init__(self):
            super().__init__(
                nn.Flatten(1),
                fc.FullyConnected(
                    layer_sizes=[obs_dim, cfg.hidden_dim, cfg.hidden_dim],
                    norm_layer=None,
                    act_layer=nn.ReLU,
                    final_layer="fc",
                ),
                dh.Beta(cfg.hidden_dim, env_f.act_space),
            )

    actor = Actor().to(device)
    actor_opt = cfg.opt.make()(actor.parameters())

    class TrainAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).sample()
            return env_f.move_act(act, to="env")

    train_agent = TrainAgent()
    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    env_iter = iter(rollout.steps(train_envs, train_agent))

    should_opt = make_sched(cfg.opt_sched)
    should_log = make_sched(cfg.log_sched)

    class ValAgent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            obs = env_f.move_obs(obs)
            act = actor(obs).mode
            return env_f.move_act(act, to="env")

    val_agent = ValAgent()
    val_envs = env_f.vector_env(cfg.val.envs, mode="val")
    val_iter_fn = lambda: rollout.episodes(
        val_envs, val_agent, max_episodes=cfg.val.episodes
    )

    sampler = data.UniformSampler()
    buf = env_f.step_buffer(cfg.buf_cap, sampler)
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)
    # prof = profiler(exp.dir / "traces", device)
    # prof.start()

    should_val = make_sched(cfg.val.sched)

    alpha = Alpha(cfg.alpha, env_f.act_space)

    pbar = tqdm(desc="Warmup", total=cfg.warmup, initial=env_step)
    rand_agent = gym.vector.agents.RandomAgent(train_envs)
    rand_iter = iter(rollout.steps(train_envs, rand_agent))

    while env_step <= cfg.warmup:
        env_idx, step = next(rand_iter)
        ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
        env_step += 1
        pbar.update(1)

    pbar = tqdm(desc="SAC", total=cfg.total_steps, initial=env_step)
    while env_step <= cfg.total_steps:
        while should_val:
            val_iter = tqdm(
                val_iter_fn(),
                desc="Val",
                total=cfg.val.episodes,
                leave=False,
            )
            val_rets = [sum(ep.reward) for _, ep in val_iter]
            exp.add_scalar("val/mean_ret", np.mean(val_rets))

        for _ in range(1):
            env_idx, step = next(env_iter)
            ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
            ep_rets[env_idx] += step.reward
            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                del ep_rets[env_idx]

            env_step += 1
            pbar.update(1)

        while should_opt:
            for _ in range(cfg.q_opt_iters):
                idxes = sampler.sample(cfg.batch_size)
                batch = env_f.fetch_step_batch(buf, idxes)

                with torch.no_grad():
                    next_act_rv = actor(batch.next_obs)
                    next_act = next_act_rv.sample()
                    min_q = torch.min(
                        qf_t[0](batch.next_obs, next_act),
                        qf_t[1](batch.next_obs, next_act),
                    )
                    next_v = min_q - alpha.value * next_act_rv.log_prob(next_act)
                    gamma = (1.0 - batch.term.float()) * cfg.gamma
                    q_targ = batch.reward + gamma * next_v

                qf0_pred = qf[0](batch.obs, batch.act)
                qf1_pred = qf[1](batch.obs, batch.act)
                q_loss = F.mse_loss(qf0_pred, q_targ) + F.mse_loss(qf1_pred, q_targ)

                qf_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                qf_opt.step()
                while should_step_polyak:
                    polyak.update(qf, qf_t, cfg.polyak.tau)

                opt_step += 1

            act_rv = actor(batch.obs)
            act = act_rv.rsample()
            actor_loss = -(
                qf[0](batch.obs, act) - alpha.value * act_rv.log_prob(act)
            ).mean()

            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()

            metrics = {}
            if alpha.adaptive:
                alpha.opt_step(act_rv.entropy(), metrics=metrics)

            while should_log:
                exp.add_scalar("train/mean_q", qf0_pred.mean())
                exp.add_scalar("train/q_loss", q_loss)
                exp.add_scalar("train/actor_loss", actor_loss)
                for k, v in metrics.items():
                    exp.add_scalar(f"train/{k}", v)


if __name__ == "__main__":
    main()
