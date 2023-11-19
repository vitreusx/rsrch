from dataclasses import dataclass
from numbers import Number

import numpy as np
import torch
from torch import Tensor, nn

from rsrch.rl import gym

from .utils import Optim


@dataclass
class Config:
    adaptive: bool
    value: float | None
    target_ent: float | None
    opt: Optim


def max_ent(act_space: gym.TensorSpace) -> float:
    if isinstance(act_space, gym.spaces.TensorDiscrete):
        return np.log(act_space.n)
    elif isinstance(act_space, gym.spaces.TensorBox):
        return torch.log(act_space.high - act_space.low).sum().item()
    else:
        raise ValueError(type(act_space))


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(self, cfg: Config, act_space: gym.TensorSpace):
        super().__init__()
        self.cfg = cfg
        if cfg.adaptive:
            self.log_alpha = nn.Parameter(torch.zeros([]))
            self.opt = cfg.opt.make()([self.log_alpha])
            self.target_ent = max_ent(act_space) * cfg.target_ent
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.value

    def opt_step(self, ent: Tensor, w: Tensor | None = None):
        """Optimize alpha value based on current entropy estimates.
        :param ent: Tensor of shape (N,) containing entropy estimates.
        :param w: Optional tensor of shape (N,) containing weights for each
        entropy value.
        :return: Either (weighted) loss values, if alpha parameter is adaptive,
        or None."""

        if self.cfg.adaptive:
            loss = self.log_alpha * (ent - self.target_ent)
            if w is not None:
                loss = loss * w
            self.opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            self.opt.step()
            self.alpha = self.log_alpha.exp().item()
            return loss

    @property
    def value(self) -> Number:
        return self.alpha
