import re
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
from rsrch.nn import dh
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from ..common import nets, plasticity
from ..common.config import MakeSched, Sched
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import autocast, find_class, null_ctx
from . import _alpha as alpha


@dataclass
class Config:
    @dataclass
    class Actor:
        encoder: dict
        dist: dict
        opt: dict

    @dataclass
    class Critic:
        encoder: dict
        dist: dict
        opt: dict

    actor: Actor
    critic: Critic
    target_critic: dict | None
    rew_norm: dict
    actor_grad: Literal["dynamics", "reinforce", "auto"]
    clip_grad: float | None
    gamma: float
    gae_lambda: float
    actor_grad_mix: float
    alpha: alpha.Config


class Actor(nn.Module):
    def __init__(
        self,
        cfg: Config.Actor,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        self.enc = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            enc_size: int = self.enc(obs).shape[1]

        layer_ctor = partial(nn.Linear, enc_size)
        self.head = dh.make(layer_ctor, act_space, **cfg.dist)

    def forward(self, obs):
        return self.head(self.enc(obs))


class Critic(nn.Module):
    def __init__(
        self,
        cfg: Config.Critic,
        obs_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space

        self.enc = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            enc_size: int = self.enc(obs).shape[1]

        layer_ctor = partial(nn.Linear, enc_size)
        vf_space = spaces.torch.Box((), dtype=torch.float32)
        self.head = dh.make(layer_ctor, vf_space, **cfg.dist)

    def forward(self, obs):
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
    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        compute_dtype: torch.dtype | None,
        make_sched: MakeSched | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor

        self.critic = self._make_critic()

        if self.cfg.target_critic is not None:
            self.target_critic = self._make_critic()
            polyak.sync(self.critic, self.target_critic)
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

        self.actor_opt = self._make_opt(self.actor.parameters(), cfg.actor.opt)
        self.critic_opt = self._make_opt(self.critic.parameters(), cfg.critic.opt)

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)

        device = next(actor.parameters()).device
        self.alpha = alpha.Alpha(cfg.alpha, actor.act_space, device, make_sched)

        self._actor_ref = plasticity.save_ref_state(self.actor)
        self._critic_ref = plasticity.save_ref_state(self.critic)

    def _make_critic(self):
        critic = Critic(self.cfg.critic, self.actor.obs_space)
        critic = critic.to(self.device)
        return critic

    def save(self):
        state = {
            "critic": self.critic.state_dict(),
            "critic_ref": self._critic_ref,
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha": self.alpha.save(),
        }

        if self.cfg.target_critic is not None:
            state = {
                **state,
                "target_critic": self.target_critic.state_dict(),
                "update_target": self.update_target.state_dict(),
            }

        return state

    def load(self, state):
        self.critic.load_state_dict(state["critic"])
        self._critic_ref = state["critic_ref"]
        self._critic_ref = {k: v.to(self.device) for k, v in self._critic_ref.items()}
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha.load(state["alpha"])

        if self.cfg.target_critic is not None:
            self.target_critic.load_state_dict(state["target_critic"])
            self.update_target.load_state_dict(state["update_target"])

    def _make_opt(self, parameters: list[nn.Parameter], cfg: dict):
        cfg = {**cfg}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        opt = cls(parameters, **cfg)
        opt = ScaledOptimizer(opt)
        return opt

    def opt_step(self, batch: Slices):
        losses = {}

        # For reinforce, we detach `target` variable, so requiring gradients on
        # any computation leading up to it will cause autograd to leak memory
        no_grad = torch.no_grad if self.actor_grad == "reinforce" else null_ctx
        with no_grad():
            with self.autocast():
                gamma = self.cfg.gamma * (1.0 - batch.term.float())
                vt = over_seq(self.target_critic)(batch.obs).mode

                target = gae_lambda(
                    reward=batch.reward[:-1],
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
            policies = over_seq(self.actor)(batch.obs[:-2].detach())
            if self.actor_grad == "dynamics":
                objective = target[1:]
            elif self.actor_grad == "reinforce":
                baseline = over_seq(self.target_critic)(batch.obs[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(batch.act[:-1].detach())
            elif self.actor_grad == "both":
                baseline = over_seq(self.target_critic)(batch.obs[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(batch.act[:-1].detach())
                mix = self.cfg.actor_grad_mix
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.alpha.value
            policy_ent = policies.entropy()
            objective = objective + ent_scale * policy_ent
            actor_loss = -(weight[:-2] * objective).mean()

        self.actor_opt.step(actor_loss, self.cfg.clip_grad)
        self.alpha.opt_step(policy_ent)

        with self.autocast():
            value_dist = over_seq(self.critic)(batch.obs[:-1].detach())
            critic_losses = -value_dist.log_prob(target.detach())
            critic_loss = (weight[:-1] * critic_losses).mean()

        self.critic_opt.step(critic_loss, self.cfg.clip_grad)
        if self.cfg.target_critic is not None:
            self.update_target.step()

        with torch.no_grad():
            with self.autocast():
                metrics = {}
                metrics["reward_mean"] = batch.reward.mean()
                metrics["reward_std"] = batch.reward.std()
                metrics["target_mean"] = target.mean()
                metrics["target_std"] = target.std()
                metrics["ent_scale"] = ent_scale
                metrics["entropy"] = policy_ent.mean()
                metrics["value_mean"] = (value_dist.mean).mean()
                metrics["actor_loss"] = actor_loss
                metrics["critic_loss"] = critic_loss

                if self.actor_grad == "both":
                    metrics["actor_grad_mix"] = mix

        return metrics

    def check_plasticity(self, batch: Slices):
        obs = batch.obs.flatten(0, 1)

        _, actor_res = plasticity.full_test(
            module=self.actor,
            ref_state=self._actor_ref,
            input=(obs,),
        )
        actor_mets = plasticity.full_metrics(actor_res)

        _, critic_res = plasticity.full_test(
            module=self.critic,
            ref_state=self._critic_ref,
            input=(obs,),
        )
        critic_mets = plasticity.full_metrics(critic_res)

        metrics = {
            **{f"actor/{k}": v for k, v in actor_mets.items()},
            **{f"critic/{k}": v for k, v in critic_mets.items()},
        }
        results = {"actor": actor_res, "critic": critic_res}

        return metrics, results

    def reset(
        self,
        make_actor: Callable[[], Actor],
        match_params: str,
        shrink_coef: float,
    ):
        match_re = re.compile(match_params)

        old_params = {
            **dict(self.actor.named_parameters(prefix="actor")),
            **dict(self.critic.named_parameters(prefix="critic")),
        }

        new_actor = make_actor()
        new_critic = self._make_critic()
        new_params = {
            **dict(new_actor.named_parameters(prefix="actor")),
            **dict(new_critic.named_parameters(prefix="critic")),
        }

        for name in old_params:
            if match_re.match(name) is not None and name in new_params:
                polyak.update_param(new_params[name], old_params[name], shrink_coef)

        polyak.sync(self.critic, self.target_critic)
