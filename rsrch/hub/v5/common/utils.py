from dataclasses import dataclass
from functools import cache, partial, wraps
from typing import Callable, Literal, ParamSpec, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl.utils import polyak


@cache
def _over_seq(_func):
    @wraps(_func)
    def _lifted(*args, **kwargs):
        batch_size = args[0].shape[1]
        y = _func(*(x.flatten(0, 1) for x in args), **kwargs)
        y = y.reshape(-1, batch_size, *y.shape[1:])
        return y

    return _lifted


P = ParamSpec("P")
R = TypeVar("R")


def over_seq(_func: Callable[P, R]) -> Callable[P, R]:
    """Transform a function that operates on batches (N, ...) to operate on
    sequences (L, N, ...). The reshape takes place for positional arguments."""
    return _over_seq(_func)


@dataclass
class Optim:
    type: Literal["adam"]
    lr: float
    eps: float

    def make(self):
        return partial(torch.optim.Adam, lr=self.lr, eps=self.eps)


@dataclass
class Polyak:
    period: int
    tau: Literal["copy"] | float

    def make(self):
        tau = 0.0 if self.tau == "copy" else self.tau
        return partial(polyak.Polyak, every=self.period, tau=tau)


def flat(x: Tensor) -> Tensor:
    return x.flatten(0, 1)
