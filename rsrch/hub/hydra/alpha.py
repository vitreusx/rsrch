from dataclasses import dataclass
from numbers import Number
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp.api import Experiment
from rsrch.rl import gym

from .utils import Optim


@dataclass
class Config:
    adaptive: bool
    opt: Optim | None = None
    value: float | None = None
    target_ent: float | Literal["auto"] | None = None
    min_value: float | None = 1e-6


def max_ent(act_space: spaces.torch.Space) -> float:
    """Compute maximum entropy for a policy over a given action space."""
    if isinstance(act_space, spaces.torch.Discrete):
        return np.log(act_space.n)
    elif isinstance(act_space, spaces.torch.Box):
        return torch.log(act_space.high - act_space.low).sum().item()
    else:
        raise ValueError(type(act_space))


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(self, cfg: Config, act_space: spaces.torch.Space):
        super().__init__()
        self.cfg = cfg
        self.adaptive = cfg.adaptive
        if cfg.adaptive:
            log_min_value = torch.tensor(cfg.min_value).log()
            self.log_alpha = nn.Parameter(torch.tensor(log_min_value))
            self.register_buffer("min_value", log_min_value)
            self.opt = cfg.opt.make()([self.log_alpha])
            self.alpha = self.log_alpha.exp().item()

            assert cfg.target_ent is not None
            if cfg.target_ent == "auto":
                if isinstance(act_space, spaces.torch.Box):
                    target_ent = -1.0
                elif isinstance(act_space, spaces.torch.Discrete):
                    target_ent = 0.75
            else:
                target_ent = cfg.target_ent
            self.target_ent = max_ent(act_space) * target_ent
        else:
            self.alpha = cfg.value

    def opt_step(
        self,
        ent: Tensor,
        w: Tensor | None = None,
        metrics: dict | None = None,
    ):
        """Optimize alpha value based on current entropy estimates.
        :param ent: Tensor of shape (N,) containing entropy estimates.
        :param w: Optional tensor of shape (N,) containing weights for each entropy value.
        :param metrics: Optional dict. If passed, auxiliary metrics are computed
        and inserted into the dict.
        :return: The loss value (or None if alpha is not adaptive).
        """

        if self.cfg.adaptive:
            loss = self.log_alpha * (ent - self.target_ent).detach()
            if w is not None:
                loss = loss * w
            self.opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            self.opt.step()
            with torch.no_grad():
                self.log_alpha.fill_(torch.max(self.log_alpha, self.min_value))
            self.alpha = self.log_alpha.exp().item()

            if metrics is not None:
                metrics["alpha"] = self.alpha
                metrics["alpha_loss"] = loss.mean()
                metrics["policy_ent"] = ent.mean()

            return loss

    @property
    def value(self) -> Number:
        return self.alpha

    def __float__(self):
        return self.alpha
