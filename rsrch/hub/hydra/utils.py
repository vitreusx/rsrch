from dataclasses import dataclass
from functools import cache, partial, singledispatchmethod, wraps
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
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


def layer_init(layer, std: float = torch.nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
    return layer


class TruncNormal2(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_space: spaces.torch.Box,
    ):
        super().__init__()
        self._out_shape = act_space.shape
        self.register_buffer("low", act_space.low)
        self.register_buffer("high", act_space.high)
        self.register_buffer("_loc", 0.5 * (act_space.low + act_space.high))
        self.register_buffer("_scale", 0.5 * (act_space.high - act_space.low))
        act_dim = int(np.prod(act_space.shape))
        self.loc_fc = nn.Linear(in_features, act_dim)
        self.loc_fc.apply(partial(layer_init, std=1e-2))
        self.log_std = nn.Parameter(torch.zeros(1, *self._out_shape))

    def forward(self, x: Tensor):
        loc: Tensor = self.loc_fc(x).reshape(-1, *self._out_shape)
        loc = 5 * loc.tanh()
        loc = self._loc + loc * self._scale
        scale = F.softplus(self.log_std) * self._scale
        rv = D.Normal(loc, scale, len(self._out_shape))
        rv = D.TruncNormal(rv, self.low, self.high)
        return rv


class GenAdvEst:
    """Generalized advantage estimator."""

    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def ret(self, rew: Tensor, term: Tensor, v: Tensor):
        """Compute amortized returns.
        :param rew: Reward tensor, of shape (L, ...).
        :param cont: Whether next state (corresponding to next_v) is
        non-terminal, of shape (L, ...).
        :param v: Value tensor, of shape (L + 1, ...)."""

        L = len(rew)
        ret = torch.empty_like(rew)
        for t in reversed(range(L)):
            if t == L - 1:
                cont = 1.0 - term.float()
                gae_est = rew[t] + self.gamma * cont * v[t + 1]
            else:
                gae_est = rew[t] + self.gamma * last_ret
            ret[t] = last_ret = (
                self.gae_lambda * gae_est + (1.0 - self.gae_lambda) * v[t]
            )

        return ret

    def __call__(self, v: Tensor, next_v: Tensor, term: Tensor, rew: Tensor):
        """Compute advantages and returns via GAE.
        :param v: Value tensor, of shape (L, ...).
        :param next_v: Next-value tensor, of shape (L, ...).
        :param cont: Whether final next-state (corresponding to next_v) is
        non-terminal, of shape (...).
        :param rew: Reward tensor, of shape (L, ...).
        """

        last_adv = 0.0
        L = len(v)
        adv = torch.empty_like(v)
        for t in reversed(range(L)):
            if t == L - 1:
                cont = 1.0 - term.float()
                delta = rew[t] + self.gamma * cont * next_v[t] - v[t]
                adv[t] = last_adv = (
                    delta + self.gamma * self.gae_lambda * cont * last_adv
                )
            else:
                delta = rew[t] + self.gamma * next_v[t] - v[t]
                adv[t] = last_adv = delta + self.gamma * self.gae_lambda * last_adv

        ret = adv + v
        return adv, ret
