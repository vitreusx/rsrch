from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing.synchronize import Lock
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.hub.rl.dreamer.common import dh
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from ..common import nets
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import autocast, find_class, null_ctx, tf_init


@dataclass
class Config:
    @dataclass
    class Actor:
        encoder: dict
        dist: dict

    @dataclass
    class Critic:
        encoder: dict
        dist: dict

    actor: Actor
    critic: Critic
    opt: dict
    coef: dict
    target_critic: dict | None
    rew_norm: dict
    actor_grad: Literal["dynamics", "reinforce", "auto"]
    clip_grad: float | None
    gamma: float
    gae_lambda: float
    actor_grad_mix: float | dict
    actor_ent: float | dict


class Actor(nn.Module):
    def __init__(
        self,
        cfg: Config.Actor,
        obs_space: spaces.torch.Tensor | spaces.torch.Tensorlike,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        if isinstance(obs_space, spaces.torch.Tensorlike):
            obs_space = obs_space.as_tensor

        self.enc = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            enc_size: int = self.enc(obs).shape[1]

        layer_ctor = partial(nn.Linear, enc_size)
        self.head = dh.make(layer_ctor, act_space, **cfg.dist)

        self.apply(tf_init)

    def forward(self, obs):
        if not isinstance(obs, Tensor):
            obs = obs.as_tensor()
        return self.head(self.enc(obs))


class Critic(nn.Module):
    def __init__(
        self,
        cfg: Config.Critic,
        obs_space: spaces.torch.Tensor | spaces.torch.Tensorlike,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space

        if isinstance(obs_space, spaces.torch.Tensorlike):
            obs_space = obs_space.as_tensor

        self.enc = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            enc_size: int = self.enc(obs).shape[1]

        layer_ctor = partial(nn.Linear, enc_size)
        vf_space = spaces.torch.Box((), dtype=torch.float32)
        self.head = dh.make(layer_ctor, vf_space, **cfg.dist)

        self.apply(tf_init)

    def forward(self, obs):
        if not isinstance(obs, Tensor):
            obs = obs.as_tensor()
        return self.head(self.enc(obs))


def gae_lambda(
    reward: Tensor,
    val: Tensor,
    gamma: Tensor,
    bootstrap: Tensor,
    lambda_: float,
):
    next_values = torch.cat((val[1:], bootstrap[None]), 0)
    inputs = reward + (1.0 - lambda_) * gamma * next_values

    returns, cur = [], val[-1]
    for t in reversed(range(len(inputs))):
        cur = inputs[t] + lambda_ * gamma[t] * cur
        returns.append(cur)

    returns.reverse()
    return torch.stack(returns)


TrainerOutput = namedtuple("TrainerOutput", ("loss", "metrics"))


class Trainer(TrainerBase):
    ON_POLICY = False

    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        make_critic: Callable[[], nn.Module],
        compute_dtype: torch.dtype | None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg

        self.actor = actor
        self.critic = make_critic()

        if self.cfg.target_critic is not None:
            self.target_critic = make_critic()
            # polyak.sync(self.critic, self.target_critic)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.target_critic,
                **self.cfg.target_critic,
            )
        else:
            self.target_critic = self.critic

        if self.cfg.actor_grad == "auto":
            discrete = isinstance(actor.act_space, spaces.torch.OneHot)
            self.actor_grad = "reinforce" if discrete else "dynamics"
        else:
            self.actor_grad = self.cfg.actor_grad

        self.opt = self._make_opt()
        self.opt_iter = 0

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)
        self.actor_grad_mix = self._make_sched(cfg.actor_grad_mix)
        self.actor_ent = self._make_sched(cfg.actor_ent)

        self._p0 = [p.data.clone() for p in self.opt.parameters]

    def save(self):
        state = super().save()
        state["opt_iter"] = self.opt_iter
        return state

    def load(self, state):
        super().load(state)
        self.opt_iter = state["opt_iter"]

    def _make_opt(self):
        cfg = {**self.cfg.opt}

        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]

        actor, critic = cfg["actor"], cfg["critic"]
        del cfg["actor"]
        del cfg["critic"]

        opt = cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.critic.parameters(), **critic},
            ],
            **cfg,
        )
        return ScaledOptimizer(opt)

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, float):
            return sched.Constant(cfg)
        else:
            cfg = {**cfg}
            cls = getattr(sched, cfg["type"])
            del cfg["type"]
            return cls(**cfg)

    def compute(self, seq: Slices):
        losses, mets = {}, {}

        # For reinforce, we detach `target` variable, so requiring gradients on
        # any computation leading up to it will cause autograd to leak memory
        no_grad = torch.no_grad if self.actor_grad == "reinforce" else null_ctx
        with no_grad():
            with self.autocast():
                gamma = self.cfg.gamma * (1.0 - seq.term.float())
                vt = over_seq(self.target_critic)(seq.obs).mode
                reward = torch.cat([torch.zeros_like(seq.reward[:1]), seq.reward])

                target = gae_lambda(
                    reward=reward[:-1],
                    val=vt[:-1],
                    gamma=gamma[:-1],
                    bootstrap=vt[-1],
                    lambda_=self.cfg.gae_lambda,
                )

        with torch.no_grad():
            with self.autocast():
                weight = torch.cat([torch.ones_like(gamma[:1]), gamma[:-1]])
                weight = weight.cumprod(0)

        with self.autocast():
            policies = over_seq(self.actor)(seq.obs[:-2].detach())
            if self.actor_grad == "dynamics":
                objective = target[1:]
            elif self.actor_grad == "reinforce":
                baseline = over_seq(self.target_critic)(seq.obs[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(seq.act[:-1].detach())
            elif self.actor_grad == "both":
                baseline = over_seq(self.target_critic)(seq.obs[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(seq.act[:-1].detach())
                mix = self.actor_grad_mix(self.opt_iter)
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.actor_ent(self.opt_iter)
            policy_ent = policies.entropy()
            objective = objective + ent_scale * policy_ent
            losses["actor"] = -(weight[:-2] * objective).mean()

            value_dist = over_seq(self.critic)(seq.obs[:-1].detach())
            critic_losses = -value_dist.log_prob(target.detach())
            losses["critic"] = (weight[:-1] * critic_losses).mean()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        with torch.no_grad():
            with self.autocast():
                mets = {}

                mets["reward_mean"] = seq.reward.mean()
                mets["reward_std"] = seq.reward.std()
                mets["target_mean"] = target.mean()
                mets["target_std"] = target.std()
                mets["ent_scale"] = ent_scale
                mets["entropy"] = policy_ent.mean()
                mets["value_mean"] = (value_dist.mean).mean()
                if "mix" in locals():
                    mets["mix"] = mix

                mets["loss"] = loss
                for k, v in losses.items():
                    mets[f"{k}_loss"] = v.detach()

        return TrainerOutput(loss, mets)

    def opt_step(self, loss: Tensor):
        self.opt.step(loss, self.cfg.clip_grad)
        self.opt_iter += 1
        if self.update_target is not None:
            self.update_target.step()

    @torch.no_grad()
    def compute_stats(self):
        p_norms, rel_p_norms, dp_norms, rel_dp_norms = [], [], [], []
        for p0, p in zip(self._p0, self.opt.parameters):
            p_norm = torch.linalg.norm(p.data)
            p_norms.append(p_norm)
            dp_norm = torch.linalg.norm(p.data - p0)
            dp_norms.append(dp_norm)
            p0_norm = torch.linalg.norm(p0)
            if p0_norm != 0.0:
                rel_p_norms.append(p_norm / p0_norm)
                rel_dp_norms.append(dp_norm / p0_norm)

        stats = {}
        stats["p_norm"] = torch.mean(torch.stack(p_norms))
        stats["dp_norm"] = torch.mean(torch.stack(dp_norms))
        if len(rel_p_norms) > 0:
            stats["rel_p_norm"] = torch.mean(torch.stack(rel_p_norms))
            stats["rel_dp_norm"] = torch.mean(torch.stack(rel_dp_norms))

        return stats


class Agent(gym.VecAgent):
    def __init__(
        self,
        actor: Actor,
        sample: bool = True,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.actor = actor
        self.obs_space = actor.obs_space
        self.act_space = actor.act_space
        self.sample = sample
        self.compute_dtype = compute_dtype
        self._obs = None

    @cached_property
    def device(self):
        return next(self.actor.parameters()).device

    @contextmanager
    def compute_ctx(self):
        if self.actor.training:
            self.actor.eval()

        with torch.no_grad():
            with autocast(self.device, self.compute_dtype):
                yield

    def reset(self, idxes, obs: Tensor):
        obs = obs.to(self.device)
        if self._obs is None:
            self._obs = obs.clone()
        else:
            self._obs[idxes] = obs.type_as(self._obs)

    def policy(self, idxes):
        with self.compute_ctx():
            policy: D.Distribution = self.actor(self._obs[idxes])
            act = policy.sample() if self.sample else policy.mode
        return act

    def step(self, idxes, act: Tensor, next_obs: Tensor):
        next_obs = next_obs.to(self.device)
        self._obs[idxes] = next_obs
