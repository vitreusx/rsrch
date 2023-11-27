from dataclasses import dataclass
from numbers import Number

import numpy as np
import torch
from torch import Tensor, nn

from rsrch.exp.api import Experiment
from rsrch.rl import gym

from .utils import Optim


@dataclass
class Config:
    adaptive: bool
    value: float | None
    target_ent: float | None
    opt: Optim


def NoAction(cb):
    ...


def max_ent(act_space: gym.TensorSpace) -> float:
    """Compute maximum entropy for a policy over a given action space."""
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
            assert cfg.target_ent is not None
            self.target_ent = max_ent(act_space) * cfg.target_ent
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.value

    def opt_step(
        self,
        ent: Tensor,
        w: Tensor | None = None,
        return_metrics=False,
    ):
        """Optimize alpha value based on current entropy estimates.
        :param ent: Tensor of shape (N,) containing entropy estimates.
        :param w: Optional tensor of shape (N,) containing weights for each entropy value.
        :param return_metrics: Whether to compute and return the metrics
        :return: A tuple (loss, metrics) with loss value (or None if alpha is not adaptive), and metrics (or empty dict if return_metrics is False)
        """

        if self.cfg.adaptive:
            loss = self.log_alpha * (ent - self.target_ent).detach()
            if w is not None:
                loss = loss * w
            self.opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            self.opt.step()
            self.alpha = self.log_alpha.exp().item()

            metrics = {}
            if return_metrics:
                metrics["train/alpha"] = self.alpha
                metrics["train/alpha_loss"] = loss.mean()
                metrics["train/policy_ent"] = ent.mean()

            return loss, metrics
        else:
            return None, {}

    @property
    def value(self) -> Number:
        return self.alpha

    def __float__(self):
        return self.alpha
