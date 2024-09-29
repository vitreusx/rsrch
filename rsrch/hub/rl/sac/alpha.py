from dataclasses import dataclass
from numbers import Number
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces

from .utils import to_camel_case


@dataclass
class Config:
    adaptive: bool
    init_value: float = 1e-8
    amp_rate: float = 0.0
    decay_rate: float = 0.0
    rel_min_ent: float | Literal["auto"] | None = None


def max_ent(act_space: spaces.torch.Tensor) -> float:
    """Compute maximum entropy for a policy over a given action space."""
    if isinstance(act_space, spaces.torch.Discrete):
        return np.log(act_space.n)
    elif isinstance(act_space, spaces.torch.Box):
        return torch.log(act_space.high - act_space.low).sum().item()
    else:
        raise ValueError(type(act_space))


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(self, cfg: Config, act_space: spaces.torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.adaptive = cfg.adaptive

        if self.adaptive:
            self.value = self.cfg.init_value
            if cfg.rel_min_ent == "auto":
                if isinstance(act_space, spaces.torch.Box):
                    rel_min_ent = -1.0
                elif isinstance(act_space, spaces.torch.Discrete):
                    rel_min_ent = 0.75
            else:
                rel_min_ent = cfg.rel_min_ent
            self.min_ent = rel_min_ent * max_ent(act_space)
        else:
            self.value = cfg.init_value

    def opt_step(self, entropy: Tensor):
        """Optimize alpha value based on current entropy estimates.
        :param entropy: Tensor of shape (N,) containing entropy estimates.
        :param metrics: Optional dict. If passed, auxiliary metrics are computed
        and inserted into the dict.
        :return: The loss value (or None if alpha is not adaptive).
        """

        if self.adaptive:
            mean_ent = entropy.mean().item()
            if mean_ent > self.min_ent:
                self.value /= 1.0 + self.cfg.decay_rate
            else:
                self.value *= 1.0 + self.cfg.amp_rate

            with torch.no_grad():
                metrics = {}
                metrics["value"] = self.value
                metrics["entropy"] = mean_ent

            return metrics

    def __float__(self):
        return self.value
