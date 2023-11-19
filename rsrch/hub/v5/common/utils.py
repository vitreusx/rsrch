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


@torch.compile
def gae_adv_est(
    rew: Tensor,
    val: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Perform Generalized Advantage Estimation (GAE).
    :param rew: Tensor of shape (L, N) containing rewards-upon-reaching for states.
    To be specific, rew[t] denotes batch of rewards upon reaching state s[t+1].
    :param val: Tensor of shape (L+1, N) containing values at given states. For
    terminal or post-terminal time-steps, values should be equal to zero.
    :return: A tuple of advantage estimates and cumulative returns, of shapes
    (L, N) each.
    """

    delta = (rew + gamma * val[1:]) - val[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, rew.shape[0])):
        adv_t = delta[t - 1] + gamma * gae_lambda * adv[-1]
        adv.append(adv_t)
    adv.reverse()
    adv = torch.stack(adv)
    ret = val[:-1] + adv
    return adv, ret
