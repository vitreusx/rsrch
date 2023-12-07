import math
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from ruamel import yaml
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.nn import dist_head as dh
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.types import Tensorlike
from rsrch.utils import _config, cron
from rsrch.utils.stats import RunningMeanStd

from . import alpha, env, utils
from .utils import Optim, gae_adv_est, layer_init


@dataclass
class Config:
    alpha: alpha.Config
    device: Literal["cuda", "cpu"]
    env_batch: int
    num_envs: int
    opt_iters: int
    opt_batch: int
    warmup: int
    total_steps: int
    clip_coeff: float
    clip_vloss: bool
    adv_norm: bool
    vf_coeff: float
    clip_grad: float | None
    gamma: float
    gae_lambda: float
    seed: int
    clip_rew: float | None
    anneal_lr: bool
    opt: Optim
    share_enc: bool


class RewardNorm:
    """Normalize rewards so that the state values have a fixed expected value."""

    def __init__(self, gamma=0.99, eps=1e-8):
        self.rms = RunningMeanStd(shape=())
        self.ret = 0.0
        self.gamma = gamma
        self.eps = eps

    def reset(self):
        self.ret = 0.0

    def update(self, rew: float):
        self.ret = self.ret * self.gamma + rew
        self.rms.update([rew])
        return rew / (float(self.rms.std) + self.eps)


class Normal2(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_space: spaces.torch.Box,
    ):
        super().__init__()
        self._out_shape = act_space.shape
        self.register_buffer("loc", 0.5 * (act_space.low + act_space.high))
        self.register_buffer("scale", 0.5 * (act_space.high - act_space.low))
        act_dim = int(np.prod(self._out_shape))
        self.mean_fc = nn.Linear(in_features, act_dim)
        self.mean_fc.apply(partial(layer_init, std=1e-2))
        self.log_std = nn.Parameter(torch.zeros(1, *self._out_shape))

    def forward(self, x: Tensor) -> D.Normal:
        mean: Tensor = self.mean_fc(x).reshape(-1, *self._out_shape)
        std = self.log_std.exp().expand_as(mean)
        rv = D.Normal(mean, std, len(self._out_shape))
        rv = D.Affine(rv, self.loc, self.scale)
        return rv


def main(env_cfg: env.Config, cfg: Config):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    env_f = env.make_factory(env_cfg, device)
    assert isinstance(env_f.net_obs_space, spaces.torch.Box)
    obs_dim = int(np.prod(env_f.net_obs_space.shape))
    is_discrete = isinstance(env_f.net_act_space, spaces.torch.Discrete)
    is_visual = env_f._visual

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            z_dim = 256 if is_visual else 64

            def make_enc():
                if is_visual:
                    enc = nn.Sequential(
                        nn.Conv2d(4, 32, 8, 4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 4, 2),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, 1),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool2d((7, 7)),
                        nn.Flatten(),
                        nn.Linear(64 * 7 * 7, z_dim),
                    )
                else:
                    enc = nn.Sequential(
                        nn.Flatten(1),
                        nn.Linear(obs_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, z_dim),
                        nn.ReLU(),
                    )

                enc.apply(layer_init)
                return enc

            critic_head = nn.Sequential(
                nn.Linear(z_dim, 1),
                nn.Flatten(0),
            )
            critic_head.apply(partial(layer_init, std=1.0))

            if is_discrete:
                actor_head = dh.Categorical(z_dim, env_f.net_act_space.n)
            else:
                actor_head = Normal2(z_dim, env_f.net_act_space)
                # actor_head = dh.TruncNormal(z_dim, env_f.act_space)

            if cfg.share_enc:
                self.enc = make_enc()
                self.critic_head = critic_head
                self.actor_head = actor_head
            else:
                self.critic = nn.Sequential(make_enc(), critic_head)
                self.actor = nn.Sequential(make_enc(), actor_head)

        def forward(self, obs: Tensor, values=True):
            if cfg.share_enc:
                z = self.enc(obs)
                v = self.critic_head(z) if values else None
                return self.actor_head(z), v
            else:
                v = self.critic(obs) if values else None
                return self.actor(obs), v

    ac = ActorCritic().to(device)
    ac_opt = cfg.opt.make()(ac.parameters())

    if cfg.anneal_lr:
        sched = torch.optim.lr_scheduler.LinearLR(ac_opt, 1.0, 0.0, cfg.total_steps)

    class make_agent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            ac.eval()
            act_rv, _ = ac(env_f.move_obs(obs), values=False)
            act = act_rv.sample()
            if not is_discrete:
                act = act.clamp(env_f.net_act_space.low, env_f.net_act_space.high)
            return env_f.move_act(act, to="env")

    envs = env_f.vector_env(num_envs=cfg.num_envs)
    agent = make_agent()
    env_iter = iter(rollout.steps(envs, agent))

    alpha_ = alpha.Alpha(cfg.alpha, env_f.net_act_space)

    exp = tensorboard.Experiment(project="hydra")
    env_step, total_steps = 0, cfg.total_steps
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(total=total_steps)

    should_log = cron.Every(lambda: env_step, 128)
    should_stop = cron.Once(lambda: env_step >= total_steps)

    def incr_env_step(n=1):
        nonlocal env_step
        env_step += n
        pbar.update(n)
        for _ in range(n):
            sched.step()

    vpg_buf = data.OnlineBuffer()
    vpg_ep_ids = [None for _ in range(envs.num_envs)]
    ep_rets = [0.0 for _ in range(envs.num_envs)]
    env_rew_norms = [RewardNorm() for _ in range(envs.num_envs)]

    for _ in range(cfg.warmup):
        next(env_iter)
        incr_env_step()

    while not should_stop:
        vpg_buf.reset()
        for _ in range(cfg.env_batch):
            env_idx, step = next(env_iter)

            ep_rets[env_idx] += step.reward
            step.reward = env_rew_norms[env_idx].update(step.reward)

            vpg_ep_ids[env_idx] = vpg_buf.push(vpg_ep_ids[env_idx], step)

            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                ep_rets[env_idx] = 0.0
                env_rew_norms[env_idx].reset()

            incr_env_step()

        episodes = [*vpg_buf.values()]
        sizes = np.array([len(ep.act) for ep in episodes])

        obs = np.stack([o for ep in episodes for o in ep.obs])
        obs = env_f.move_obs(obs)
        ep_obs = obs.split_with_sizes([*(sizes + 1)])

        obs_mask = []
        for o in ep_obs:
            obs_mask.extend([True] * (len(o) - 1))
            obs_mask.append(False)
        obs_mask = torch.tensor(obs_mask)
        obs_idxes = torch.where(obs_mask)[0]

        act = np.stack([a for ep in episodes for a in ep.act])
        act = env_f.move_act(act)

        rew = torch.tensor(
            [r for ep in episodes for r in ep.reward],
            device=device,
            dtype=torch.float32,
        )

        term = [ep.term for ep in episodes]

        if cfg.clip_rew is not None:
            rew = rew.clamp(-cfg.clip_rew, cfg.clip_rew)

        with torch.no_grad():
            policy, val = ac(obs)
            logp = policy[obs_idxes].log_prob(act)

            ep_rews = rew.split_with_sizes([*sizes])
            ep_vals = val.split_with_sizes([*(sizes + 1)])

            adv, ret = [], []
            for ep_term, ep_rew, ep_val in zip(term, ep_rews, ep_vals):
                if ep_term:
                    ep_val[-1] = 0.0
                ep_adv, ep_ret = gae_adv_est(ep_rew, ep_val, cfg.gamma, cfg.gae_lambda)
                adv.append(ep_adv)
                ret.append(ep_ret)
            adv, ret = torch.cat(adv), torch.cat(ret)

            val = torch.cat([v[:-1] for v in ep_vals])

        metrics = {} if should_log else None

        for _ in range(cfg.opt_iters):
            perm = torch.randperm(len(act))
            for idxes in perm.split(cfg.opt_batch):
                idxes_ = obs_idxes[idxes]
                new_policy, new_value = ac(obs[idxes_])
                new_logp = new_policy.log_prob(act[idxes])
                log_ratio = new_logp - logp[idxes]
                ratio = log_ratio.exp()

                adv_ = adv[idxes]
                if cfg.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)

                t1 = -adv_ * ratio
                t2 = -adv_ * ratio.clamp(1 - cfg.clip_coeff, 1 + cfg.clip_coeff)
                policy_loss = torch.max(t1, t2).mean()

                if cfg.clip_vloss:
                    clipped_v = val[idxes] + (new_value - val[idxes]).clamp(
                        -cfg.clip_coeff, cfg.clip_coeff
                    )
                    v_loss1 = (new_value - ret[idxes]).square()
                    v_loss2 = (clipped_v - ret[idxes]).square()
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_value - ret[idxes]).square().mean()
                v_loss = cfg.vf_coeff * v_loss

                policy_ent = new_policy.entropy()

                ent_adv = alpha_.value * policy_ent
                t1 = -ent_adv * ratio
                t2 = -ent_adv * ratio.clamp(1 - cfg.clip_coeff, 1 + cfg.clip_coeff)
                ent_loss = torch.max(t1, t2).mean()

                loss = policy_loss + v_loss + ent_loss
                ac_opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(ac.parameters(), cfg.clip_grad)
                ac_opt.step()

                if metrics is not None:
                    if "train/loss" not in metrics:
                        metrics["train/loss"] = loss
                        metrics["train/policy_loss"] = policy_loss
                        metrics["train/v_loss"] = v_loss
                        metrics["train/ent_loss"] = ent_loss
                        metrics["train/mean_v"] = val.mean()
                        metrics["train/mean_ent"] = policy_ent.mean()

                alpha_.opt_step(policy_ent, metrics=metrics)

        if cfg.anneal_lr:
            if metrics is not None:
                metrics["train/lr"] = sched.get_last_lr()[0]

        if metrics is not None:
            for name, value in metrics.items():
                exp.add_scalar(name, value)


if __name__ == "__main__":
    main()
