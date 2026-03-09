from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@torch.no_grad()
def update_param(source: nn.Parameter, target: nn.Parameter, tau: float):
    """Update target parameter by mixing with a source parameter.
    Also known as Polyak averaging.

    :param source: Source parameter.
    :param target: Target parameter.
    :param tau: Update coefficient. Passing :math:`\\tau = 0` copies `source`
    to `target`.
    """

    new_val = tau * target.data + (1.0 - tau) * source.data
    target.data.copy_(new_val)


@torch.no_grad()
def update(source: nn.Module, target: nn.Module, tau: float):
    """Update target module by mixing with a source module.
    Also known as Polyak averaging.

    :param source: Source parameter.
    :param target: Target parameter.
    :param tau: Update coefficient. Passing :math:`\\tau = 0` copies `source`
    to `target`.
    """
    for target_p, source_p in zip(
        target.parameters(), source.parameters(), strict=False
    ):
        update_param(source_p, target_p, tau)


@torch.no_grad()
def sync(source: nn.Module, target: nn.Module):
    """Sync target module's parameters with a source module."""
    target.load_state_dict(source.state_dict())


@dataclass
class Config:
    tau: float | Literal["sync"] = 0.0
    every: int = 1


class Polyak:
    """A stateful manager for Polyak averaging."""

    def __init__(
        self,
        source: nn.Module,
        target: nn.Module,
        tau: float | Literal["sync"] = 0.0,
        every: int = 1,
    ):
        """Create a stateful manager for Polyak averaging.

        :param source: Source module.
        :param target: Target module.
        :param tau: Update coefficient, or `sync`. In the former case, every
        :math:`k` steps an `update` procedure is called with a specified value
        of `tau`. If `sync`, a simple copy is performed.
        """

        self.source = source
        self.target = target
        self.tau = tau
        self._step, self._last, self.every = 0, 0, every

    def step(self, n=1):
        self._step += n
        while self._step - self._last >= self.every:
            if self.tau == "sync":
                sync(self.source, self.target)
            else:
                update(self.source, self.target, self.tau)
            self._last += self.every

    def state_dict(self):
        return {"step": self._step, "last": self._last}

    def load_state_dict(self, state: dict):
        self._step = state["step"]
        self._last = state["last"]
