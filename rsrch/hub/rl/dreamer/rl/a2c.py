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

        self.apply(tf_init)

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

        self.apply(tf_init)

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
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor

        self.critic = self._make_critic()

        if self.cfg.target_critic is not None:
            self.target_critic = self._make_critic()
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

        self._actor_ref = plasticity.save_ref_state(self.actor)
        self._critic_ref = plasticity.save_ref_state(self.critic)

    def _make_critic(self):
        critic = Critic(self.cfg.critic, self.actor.obs_space)
        critic = critic.to(self.device)
        return critic

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

    def opt_step(self, batch: Slices):
        losses = {}

        # For reinforce, we detach `target` variable, so requiring gradients on
        # any computation leading up to it will cause autograd to leak memory
        no_grad = torch.no_grad if self.actor_grad == "reinforce" else null_ctx
        with no_grad():
            with self.autocast():
                gamma = self.cfg.gamma * (1.0 - batch.term.float())
                vt = over_seq(self.target_critic)(batch.obs).mode
                reward = torch.cat([torch.zeros_like(batch.reward[:1]), batch.reward])

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
                mix = self.actor_grad_mix(self.opt_iter)
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.actor_ent(self.opt_iter)
            policy_ent = policies.entropy()
            objective = objective + ent_scale * policy_ent
            losses["actor"] = -(weight[:-2] * objective).mean()

            value_dist = over_seq(self.critic)(batch.obs[:-1].detach())
            critic_losses = -value_dist.log_prob(target.detach())
            losses["critic"] = (weight[:-1] * critic_losses).mean()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        self.opt.step(loss, self.cfg.clip_grad)
        self.opt_iter += 1
        if self.update_target is not None:
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

                if "mix" in locals():
                    metrics["mix"] = mix

                metrics["loss"] = loss
                for k, v in losses.items():
                    metrics[f"{k}_loss"] = v.detach()

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
