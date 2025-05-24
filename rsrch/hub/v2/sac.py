from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, TypedDict

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer

import rsrch.torch.distributions as D
from rsrch.rl.utils import polyak
from rsrch.rl.utils.gae import gae_only_ret
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.torch.nn.utils import frozen, over_seq


class Policy(D.Distribution, Tensor):
    """SAC action distribution interface."""


class Actor(nn.Module):
    """SAC actor interface."""

    def __call__(self, state: Tensor) -> Policy:
        ...


class QFunc(nn.Module):
    """Interface for SAC Q functions."""

    def __call__(
        self,
        state: Tensor,
        act: Tensor | None = None,
    ) -> Tensor:
        ...

    @property
    def optimized_for_discrete(self) -> bool:
        """(Optional) Whether `__call__` supports passing only `state`, and returning value estimates for all actions. If such is the case, one can evaluate value expectation directly and reduce variance."""


class Alpha:
    """Interface for :math:`\alpha` entropy coefficient."""

    def opt_step(self, entropy: float):
        ...

    def __float__(self):
        ...


@dataclass
class Slices:
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor


class Trainer:
    def __init__(
        self,
        actor: Actor,
        make_qf: Callable[[], QFunc],
        make_actor_opt: Callable[[list[Parameter]], Optimizer],
        make_qf_opt: Callable[[list[Parameter]], Optimizer],
        num_qf: int,
        target_qf: polyak.Config,
        gamma: float,
        gae_lambda: float,
        alpha: float | Alpha,
        clip_grad: float | None,
        rew_norm: Literal["sign", "tanh"] | Callable[[Tensor], Tensor] | None,
        compute_dtype: torch.dtype,
    ):
        self.actor = actor
        self.num_critics = num_qf
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.clip_grad = clip_grad
        self.rew_norm = rew_norm
        self.compute_dtype = compute_dtype

        self.qfs, self.qf_ts = nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_critics):
            self.qfs.append(make_qf())
            self.qf_ts.append(make_qf())

        self.qf_ts.requires_grad_(False)
        polyak.sync(self.qfs, self.qf_ts)

        self.qf_polyak = polyak.Polyak(
            source=self.qfs,
            target=self.qf_ts,
            tau=target_qf.tau,
            every=target_qf.every,
        )

        self.actor_opt = make_actor_opt(self.actor.parameters())
        self.actor_opt = ScaledOptimizer(self.actor_opt, self.compute_dtype)

        self.qf_opt = make_qf_opt(self.qfs.parameters())
        self.qf_opt = ScaledOptimizer(self.qf_opt, self.compute_dtype)

        device = next(self.actor.parameters()).device
        self.autocast = lambda: torch.autocast(device.type, compute_dtype)

        self._fast_discrete = getattr(self.qfs[0], "optimized_for_discrete", False)

    def opt_step(self, batch: Slices, return_metrics: bool = False):
        obs, next_obs = batch.obs[:-1], batch.obs[1:]

        with torch.no_grad():
            cont = 1.0 - batch.term.float()
            weight = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])
            weight = torch.cumprod(weight, dim=0)

        with self.autocast():
            policy = over_seq(self.actor)(batch.obs)

        with torch.no_grad():
            with self.autocast():
                next_policy: D.Distribution = policy[1:].detach()

                if self._fast_discrete:
                    min_q = over_seq(self.qf_ts[0])(next_obs)
                    for idx in range(1, self.num_critics):
                        min_q_idx = over_seq(self.qf_ts[idx])(next_obs)
                        min_q = torch.min(min_q, min_q_idx)

                    q_values = min_q - float(self.alpha) * next_policy.log_probs
                    next_v = (next_policy.probs * q_values).sum(-1)
                else:
                    next_act = next_policy.sample()
                    min_q = over_seq(self.qf_ts[0])(next_obs, next_act)
                    for idx in range(1, self.num_critics):
                        min_q_idx = over_seq(self.qf_ts[idx])(next_obs, next_act)
                        min_q = torch.min(min_q, min_q_idx)
                    next_v = min_q - float(self.alpha) * next_policy.log_prob(next_act)

                gamma = cont * self.gamma
                reward = self._rew_norm(batch.reward)
                target = gae_only_ret(reward, next_v, gamma[1:], self.gae_lambda)

        with self.autocast():
            qf_losses = []
            for critic in self.qfs:
                qf_pred = over_seq(critic)(obs, batch.act)
                qf_loss = (weight[:-1] * (qf_pred - target).square()).mean()
                qf_losses.append(qf_loss)
            qf_loss = sum(qf_losses)

        self.qf_opt.step(qf_loss, clip_grad=self.clip_grad)

        cur_policy = policy[:-1]
        if self._fast_discrete:
            with torch.no_grad():
                with self.autocast():
                    min_q = over_seq(self.qfs[0])(obs)
                    for idx in range(1, self.num_critics):
                        min_q_idx = over_seq(self.qfs[idx])(obs)
                        min_q = torch.min(min_q, min_q_idx)

            with self.autocast():
                policy: D.Categorical
                actor_losses = float(self.alpha) * cur_policy.log_probs - min_q
                actor_losses = (cur_policy.probs * actor_losses).sum(-1)
                actor_loss = (weight[:-1] * actor_losses).mean()
        else:
            with self.autocast():
                act = cur_policy.rsample()
                with frozen(self.qfs):
                    min_q = over_seq(self.qfs[0])(obs, act)
                    for idx in range(1, self.num_critics):
                        min_q_idx = over_seq(self.qfs[idx])(obs, act)
                        min_q = torch.min(min_q, min_q_idx)
                actor_losses = float(self.alpha) * cur_policy.log_prob(act) - min_q
                actor_loss = (weight[:-1] * actor_losses).mean()

        self.actor_opt.step(actor_loss, self.clip_grad)

        if hasattr(self.alpha, "opt_step") or return_metrics:
            with torch.no_grad():
                entropy = policy.entropy().mean().item()

        if hasattr(self.alpha, "opt_step"):
            self.alpha.opt_step(entropy)

        result = {}

        if return_metrics:
            with torch.no_grad():
                result["metrics"] = {
                    "qf_loss": qf_loss / self.num_critics,
                    "mean_q": qf_pred.mean(),
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    "alpha": float(self.alpha),
                }

        return result

    def _rew_norm(self, reward: Tensor):
        if callable(self.rew_norm):
            return self.rew_norm(reward)
        elif self.rew_norm == "sign":
            return torch.sign(reward)
        elif self.rew_norm == "tanh":
            return torch.tanh(reward)
        else:
            return reward
