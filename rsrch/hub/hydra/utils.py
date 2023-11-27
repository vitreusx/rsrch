from dataclasses import dataclass
from functools import cache, partial, wraps
from typing import Literal

import torch
from torch import Tensor

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


def over_seq(_func):
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


def gae_adv_est(r: Tensor, v: Tensor, gamma: float, gae_lambda: float):
    """Compute returns and advantage estimates.
    :param r: Reward tensor, of shape (L, N).
    :param v: Value estimate tensor, of shape (L+1, N).
    :param gamma: Discount value.
    :param gae_lambda: Weight term for future traces."""
    delta = (r + gamma * v[1:]) - v[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(r))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = v[:-1] + adv
    return adv, ret
