from __future__ import annotations

from typing import Any, Callable, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer

import rsrch.distributions as D
from rsrch.nn.optim import ScaledOptimizer
from rsrch.nn.utils import over_seq
from rsrch.rl.utils import polyak
from rsrch.rl.utils.gae import gae_only_ret


class QDist:
    """Rainbow Q value dist interface."""

    def __mul__(self, scale: Tensor) -> QDist:
        ...

    def __add__(self, loc: Tensor) -> QDist:
        ...

    def gather(self, dim: int, index: Tensor) -> QDist:
        ...

    def squeeze(self, dim: int) -> QDist:
        ...

    def __getitem__(self, index: Any) -> QDist:
        ...

    def reshape(self, shape: tuple[int, ...]) -> QDist:
        ...

    def flatten(self, start_dim: int, end_dim: int) -> QDist:
        ...

    @property
    def mean(self) -> Tensor:
        ...


class QFunc(nn.Module):
    """Rainbow Q function interface."""

    def __call__(self, state: Tensor) -> Tensor | QDist:
        """Get Q value estimates :math:`Q(s_t, a_t)` for each possible action :math:`a_t`. Either a point estimate (a `torch.Tensor`), or a Q  distribution."""


class Actor:
    def __init__(self, qf: QFunc):
        self.qf = qf

    def __call__(self, state: Tensor) -> D.Distribution:
        with torch.inference_mode():
            q_values = self.qf(state)
            if not isinstance(q_values, Tensor):
                q_values = q_values.mean
            return q_values.argmax(-1)


class Batch(TypedDict):
    obs: Tensor  # (L+1, N, *O)
    act: Tensor  # (L, N, *A)
    reward: Tensor  # (L, N)
    term: Tensor  # (L+1, N)
    weight: Tensor  # Optional, (N,)


class Trainer:
    def __init__(
        self,
        qf: QFunc,
        make_qf: Callable[[], QFunc],
        make_qf_opt: Callable[[list[Parameter]], Optimizer],
        target_qf: polyak.Config,
        rew_norm: Callable[[Tensor], Tensor] | None,
        double_dqn: bool,
        gamma: float,
        gae_lambda: float,
        q_div: Callable[[QDist, QDist], Tensor] | None,
        clip_grad: float | None,
        compute_dtype: torch.dtype,
    ):
        self.qf = qf
        self.rew_norm = rew_norm
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.q_div = q_div
        self.clip_grad = clip_grad
        self.compute_dtype = compute_dtype

        self.qf_t = make_qf()
        self.qf_t.requires_grad_(False)
        polyak.sync(self.qf, self.qf_t)

        self.qf_polyak = polyak.Polyak(
            source=self.qf,
            target=self.qf_t,
            tau=target_qf.tau,
            every=target_qf.every,
        )

        self.qf_opt = make_qf_opt(self.qf.parameters())
        self.qf_opt = ScaledOptimizer(self.qf_opt, self.compute_dtype)

        device = next(self.qf.parameters()).device
        self.autocast = lambda: torch.autocast(device.type, compute_dtype)

    def opt_step(
        self,
        batch: Batch,
        compute_metrics: bool = False,
    ):
        reward = batch["reward"]
        if self.rew_norm is not None:
            reward = self.rew_norm(reward)

        with torch.no_grad():
            with self.autocast():
                next_obs = batch["obs"][1:]
                next_q_eval = over_seq(self.qf_t)(next_obs)

                if isinstance(next_q_eval, Tensor):
                    if self.double_dqn:
                        next_q_act = over_seq(self.qf)(next_obs)
                        act = next_q_act.argmax(-1)
                        target = next_q_eval.gather(-1, act[..., None])
                        target = target.squeeze(-1)
                    else:
                        target = next_q_eval.max(-1).values
                else:
                    if self.double_dqn:
                        next_q_act = over_seq(self.qf)(next_obs)
                    else:
                        next_q_act = next_q_eval
                    act = next_q_act.mean.argmax(-1)
                    target = next_q_eval.gather(-1, act[..., None])
                    target = target.squeeze(-1)

                next_gamma = self.gamma * (1.0 - batch["term"].float())
                target: Tensor | QDist = gae_only_ret(
                    reward, next_q_eval, next_gamma, self.gae_lambda
                )

        with self.autocast():
            qv = over_seq(self.qf(batch["obs"][:-1]))
            pred = qv.gather(-1, batch["act"][..., None])
            pred = pred.squeeze(-1)

            if isinstance(target, Tensor):
                q_losses = (pred - target).square()
            else:
                q_losses = self.q_div(target, pred)

            prio = q_losses.mean(0)
            if "weight" in batch:
                q_losses = batch["weight"][None] * prio

        loss = q_losses.mean()
        self.qf_opt.step(loss, self.clip_grad)

        result = {}

        if "weight" in batch:
            result["prio"] = prio.detach()

        if compute_metrics:
            with torch.no_grad():
                result["metrics"] = {
                    "loss": loss.detach(),
                    "mean_q_target": target.mean(),
                    "mean_q_pred": pred.mean(),
                }

        return result
