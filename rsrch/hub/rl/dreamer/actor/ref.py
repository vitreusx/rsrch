from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from ..common import nets
from ..common.trainer import TrainerBase
from ..common.utils import find_class, null_ctx, tf_init


@dataclass
class Config:
    actor: dict
    actor_dist: dict
    critic: dict
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


class Actor(nn.Sequential):
    def __init__(
        self,
        cfg: Config,
        state_size: int,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg

        mlp = nets.MLP(state_size, None, **cfg.actor)
        head = nets.ActorHead(mlp.out_features, act_space, **cfg.actor_dist)
        super().__init__(mlp, head)

        self.apply(tf_init)


class Critic(nets.BoxDecoder):
    def __init__(self, cfg: Config, state_size: int):
        vf_space = spaces.torch.Box((), dtype=torch.float32)
        super().__init__(state_size, vf_space, **cfg.critic)

        self.apply(tf_init)


def gae_lambda(rew, val, gamma, bootstrap, lambda_):
    next_values = torch.cat((val[1:], bootstrap[None]), 0)
    inputs = rew + (1.0 - lambda_) * gamma * next_values

    returns, cur = [], bootstrap
    for t in reversed(range(len(inputs))):
        cur = inputs[t] + lambda_ * gamma[t] * cur
        returns.append(cur)

    returns.reverse()
    return torch.stack(returns)


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        compute_dtype: torch.dtype | None,
    ):
        super().__init__(
            clip_grad=cfg.clip_grad,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)
        self.actor_grad_mix = self._make_sched(cfg.actor_grad_mix)
        self.actor_ent = self._make_sched(cfg.actor_ent)

    def setup(
        self,
        actor: Actor,
        make_critic: Callable[[], nn.Module],
        act_space: spaces.torch.Tensor,
    ):
        self.actor = actor

        self.critic = make_critic()

        if self.cfg.target_critic is not None:
            self.target_critic = make_critic()
            polyak.sync(self.critic, self.target_critic)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.target_critic,
                **self.cfg.target_critic,
            )
        else:
            self.target_critic = self.critic

        if self.cfg.actor_grad == "auto":
            discrete = isinstance(act_space, spaces.torch.Discrete)
            self.actor_grad = "reinforce" if discrete else "dynamics"
        else:
            self.actor_grad = self.cfg.actor_grad

        self.opt = self._make_opt()
        self.parameters = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt_iter = 0
        self._device = next(self.actor.parameters()).device
        self.scaler = getattr(torch, self._device.type).amp.GradScaler()

    def _make_opt(self):
        cfg = {**self.cfg.opt}

        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]

        actor, critic = cfg["actor"], cfg["critic"]
        del cfg["actor"]
        del cfg["critic"]

        common = cfg

        return cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.critic.parameters(), **critic},
            ],
            **common,
        )

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, float):
            return sched.Constant(cfg)
        else:
            cfg = {**cfg}
            cls = getattr(sched, cfg["type"])
            del cfg["type"]
            return cls(**cfg)

    def compute(self, states, action, reward, term):
        losses, mets = {}, {}

        mets["reward_mean"] = reward.mean()
        mets["reward_std"] = reward.std()

        if self.update_target is not None:
            self.update_target.step()

        # For reinforce, we detach `target` variable, so requiring gradients on
        # any computation leading up to it will cause autograd to leak memory
        no_grad = torch.no_grad if self.actor_grad == "reinforce" else null_ctx
        with no_grad():
            with self.autocast():
                gamma = self.cfg.gamma * (1.0 - term.float())

                vt = over_seq(self.target_critic)(states).mode
                target = gae_lambda(
                    rew=reward[:-1],
                    val=vt[:-1],
                    gamma=gamma[:-1],
                    bootstrap=vt[-1],
                    lambda_=self.cfg.gae_lambda,
                )

                mets["target_mean"] = target.mean()
                mets["target_std"] = target.std()

        with torch.no_grad():
            with self.autocast():
                weight = torch.cat([torch.ones_like(gamma[:1]), gamma[:-1]])
                weight = weight.cumprod(0)

        with self.autocast():
            policies = over_seq(self.actor)(states[:-2].detach())
            if self.actor_grad == "dynamics":
                objective = target[1:]
            elif self.actor_grad == "reinforce":
                baseline = over_seq(self.target_critic)(states[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(action[:-1].detach())
            elif self.actor_grad == "both":
                baseline = over_seq(self.target_critic)(states[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(action[:-1].detach())
                mix = self.actor_grad_mix(self.opt_iter)
                mets["actor_grad_mix"] = mix
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.actor_ent(self.opt_iter)
            mets["policy_ent_scale"] = ent_scale
            policy_ent = policies.entropy()
            mets["policy_ent"] = policy_ent.mean()
            objective = objective + ent_scale * policy_ent
            losses["actor"] = -(weight[:-2] * objective).mean()

            value_dist = over_seq(self.critic)(states[:-1].detach())
            mets["value"] = (value_dist.mean).mean()
            critic_losses = -value_dist.log_prob(target.detach())
            losses["critic"] = (weight[:-1] * critic_losses).mean()

            for k, v in losses.items():
                mets[f"{k}_loss"] = v.detach()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        return loss, mets
