import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ruamel import yaml
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar
from rsrch.nn import dist_head as dh
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import _config, cron

from . import alpha, env, utils
from .utils import gae_adv_est


def layer_init(layer, std=torch.nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
    return layer


@dataclass
class Config:
    env: env.Config
    alpha: alpha.Config
    device: Literal["cuda", "cpu"] = "cuda"
    """Device to use."""
    env_batch: int = 2048
    """Total number of env steps per iteration."""
    num_envs: int = 1
    """Number of parallel envs."""
    opt_iters: int = 10
    """Number of opt epochs for each env rollout."""
    opt_batch: int = 64
    """Minibatch size to use during optimization."""
    warmup: int = int(10e3)
    """Number of env steps to do before optimizing."""
    total_steps: int = int(1e6)
    """Total number of env steps."""
    clip_coeff: float = 0.2
    """PPO clip coefficient."""
    clip_vloss: bool = True
    """Whether to clip state values as well."""
    adv_norm: bool = True
    """Whether to normalize advantage values."""
    vf_coeff: float = 0.5
    """Coefficient for value loss."""
    clip_grad: float | None = 0.5
    """Whether (and to what value) to clip norm of the gradients."""
    gamma: float = 0.99
    """Discount value."""
    gae_lambda: float = 0.95
    """GAE weight coefficient."""
    seed: int = 42
    """RNG seed."""
    clip_rew: float | None = 1.0
    """Whether (and to what maximum absolute value) to clip rewards.
    Doesn't affect episode returns."""
    anneal_lr: bool = True
    """Whether to anneal learning rate. The schedule is linear to zero."""


class Normal(nn.Module):
    def __init__(self, in_features: int, act_space: gym.spaces.TensorBox):
        super().__init__()
        self._out_shape = act_space.shape
        self.register_buffer("_low", act_space.low)
        self.register_buffer("_high", act_space.high)
        out_features = int(np.prod(act_space.shape))
        self.mean_fc = nn.Linear(in_features, out_features)
        self.mean_fc.apply(partial(layer_init, std=1e-2))
        self.log_std = nn.Parameter(torch.zeros(1, *act_space.shape))

    def forward(self, x: Tensor):
        mean: Tensor = self.mean_fc(x)
        mean = mean.clamp(self._low, self._high)
        mean = mean.reshape(-1, *self._out_shape)
        std = self.log_std.expand_as(mean).exp()
        return D.Normal(mean, std, len(self._out_shape))


def main():
    cfg = _config.cli(
        Config,
        config_yml=Path(__file__).parent / "config.yml",
        presets_yml=Path(__file__).parent / "presets.yml",
        # args=["-h"],
        args=["-p", "Walker2d-v4"],
    )

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    env_f = env.make_factory(cfg.env, device)
    assert isinstance(env_f.obs_space, gym.spaces.TensorBox)
    obs_dim = int(np.prod(env_f.obs_space.shape))
    is_discrete = isinstance(env_f.act_space, gym.spaces.TensorDiscrete)

    def make_actor():
        stem = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        stem.apply(layer_init)

        if is_discrete:
            head = dh.Categorical(64, env_f.act_space.n, min_pr=1e-8)
        else:
            env_f.act_space: gym.spaces.TensorBox
            # head = dh.Normal(
            #     in_features=128,
            #     out_shape=env_f.act_space.shape,
            #     std="exp",
            # )
            # head = dh.Piecewise(
            #     in_features=64,
            #     out_shape=env_f.act_space.shape,
            #     num_buckets=16,
            #     low=env_f.act_space.low,
            #     high=env_f.act_space.high,
            # )
            head = Normal(
                in_features=64,
                act_space=env_f.act_space,
            )

        return nn.Sequential(stem, head).to(device)

    # act_enc = lambda x: nn.functional.one_hot(x.long(), act_dim).float()
    # act_dec = lambda x: x.argmax(-1).long()
    act_enc = lambda x: x
    act_dec = lambda x: x

    actor = make_actor()

    def make_critic():
        stem = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        stem.apply(layer_init)

        head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Flatten(0),
        )
        head.apply(partial(layer_init, std=1.0))

        return nn.Sequential(stem, head).to(device)

    critic = make_critic()

    ac_params = [*actor.parameters(), *critic.parameters()]
    ac_opt = torch.optim.Adam(ac_params, lr=3e-4)

    if cfg.anneal_lr:
        sched = torch.optim.lr_scheduler.LinearLR(ac_opt, 1.0, 0.0, cfg.total_steps)

    class make_agent(gym.vector.Agent):
        @torch.inference_mode()
        def policy(self, obs):
            actor.eval()
            act_rv = actor(env_f.move_obs(obs))
            actor.train()
            act = act_dec(act_rv.sample())
            return env_f.move_act(act, to="env")

    envs = env_f.vector_env(num_envs=cfg.num_envs)
    agent = make_agent()
    env_iter = iter(rollout.steps(envs, agent))

    alpha_ = alpha.Alpha(cfg.alpha, env_f.act_space)

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
        if should_log:
            exp.add_scalar("train/lr", sched.get_last_lr()[0])

    vpg_buf = data.OnlineBuffer()
    vpg_ep_ids = [None for _ in range(envs.num_envs)]

    ep_rets = [0.0 for _ in range(envs.num_envs)]

    for _ in range(cfg.warmup):
        next(env_iter)
        incr_env_step()

    while not should_stop:
        vpg_buf.reset()
        for _ in range(cfg.env_batch):
            env_idx, step = next(env_iter)
            vpg_ep_ids[env_idx] = vpg_buf.push(vpg_ep_ids[env_idx], step)

            ep_rets[env_idx] += step.reward
            if step.done:
                exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                ep_rets[env_idx] = 0.0

            incr_env_step()

        episodes = [*vpg_buf.values()]
        sizes = np.array([len(ep.act) for ep in episodes])

        obs = np.stack([o for ep in episodes for o in ep.obs])
        all_obs = env_f.move_obs(obs)
        ep_obs = all_obs.split_with_sizes([*(sizes + 1)])
        obs = torch.cat([o[:-1] for o in ep_obs])

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
            policy: D.Distribution = actor(obs)
            logp = policy.log_prob(act)
            val: Tensor = critic(all_obs)

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
                new_policy = actor(obs[idxes])
                new_value = critic(obs[idxes])
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
                ent_loss = alpha_.value * -policy_ent.mean()

                loss = policy_loss + v_loss + ent_loss
                ac_opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(ac_params, cfg.clip_grad)
                ac_opt.step()

                if metrics is not None and "train/loss" not in metrics:
                    metrics["train/loss"] = loss
                    metrics["train/policy_loss"] = policy_loss
                    metrics["train/v_loss"] = v_loss
                    metrics["train/ent_loss"] = ent_loss
                    metrics["train/mean_v"] = val.mean()

                alpha_.opt_step(policy_ent, metrics=metrics)

        if metrics is not None:
            for name, value in metrics.items():
                exp.add_scalar(name, value)


if __name__ == "__main__":
    main()
