import math
from functools import cached_property
from numbers import Number

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.distributions.utils import sum_rightmost
from rsrch.types import Tensorlike


def phi(zeta: Tensor):
    return torch.exp(-0.5 * zeta.square()) / np.sqrt(2.0 * np.pi)


def Phi(x: Tensor):
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def Phi_inv(y: Tensor, eps=1e-7):
    z = (2.0 * y - 1.0).clamp(-1.0 + eps, 1.0 - eps)
    return np.sqrt(2.0) * torch.erfinv(z)


def lazy_property(_func):
    @property
    def _prop(self):
        var = "_" + _func.__name__
        if not hasattr(self, var):
            self.register(var, None)
        if getattr(self, var) is None:
            setattr(self, var, _func(self))
        return getattr(self, var)

    return _prop


class TruncNormal(D.Distribution, Tensorlike):
    def __init__(
        self,
        norm: D.Normal,
        low: Number | Tensor,
        high: Number | Tensor,
        event_dims: int = 0,
    ):
        low = torch.as_tensor(low, device=norm.device)
        low = low.expand_as(norm.loc)
        high = torch.as_tensor(high, device=norm.device)
        high = high.expand_as(norm.loc)

        Tensorlike.__init__(self, norm.batch_shape)
        self.event_shape = norm.event_shape

        self.norm = self.register("norm", norm)
        self.low = self.register("low", low)
        self.high = self.register("high", high)

    @lazy_property
    def alpha(self):
        return self._zeta(self.low)

    @lazy_property
    def phi_alpha(self):
        return phi(self.alpha)

    @lazy_property
    def Phi_alpha(self):
        return Phi(self.alpha)

    @lazy_property
    def beta(self):
        return self._zeta(self.high)

    @lazy_property
    def phi_beta(self):
        return phi(self.beta)

    @property
    def Phi_beta(self):
        return Phi(self.beta)

    @property
    def norm_Z(self):
        return Phi(self.beta) - Phi(self.alpha)

    def _zeta(self, x):
        return (x - self.norm.loc) / self.norm.scale

    @property
    def mean(self):
        f = (self.phi_alpha - self.phi_beta) / self.norm_Z
        return self.norm.mean + f * self.norm.scale

    @property
    def mode(self):
        return self.norm.mode.clamp(self.low, self.high)

    @property
    def var(self):
        f1 = (self.beta * self.phi_beta - self.alpha * self.phi_alpha) / self.norm_Z
        f2 = ((self.phi_alpha - self.phi_beta) / self.norm_Z).square()
        return self.norm.var * (1.0 - f1 - f2)

    def entropy(self):
        norm_ent = 0.5 + 0.5 * math.log(2.0 * math.pi) + self.norm.scale.log()
        f = self.norm_Z.log() + (
            0.5
            * (self.alpha * self.phi_alpha - self.beta * self.phi_beta)
            / self.norm_Z
        )
        ent = norm_ent + f
        return sum_rightmost(ent, len(self.event_shape))

    def log_prob(self, value: Tensor):
        norm_logp = (
            -((value - self.norm.loc) ** 2) / (2 * self.norm.var)
            - self.norm.scale.log()
            - 0.5 * math.log(2 * math.pi)
        )
        logp = norm_logp - self.norm_Z.log()
        return sum_rightmost(logp, len(self.event_shape))

    def sample(self, sample_shape: torch.Size = torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        shape = torch.Size([*sample_shape, *self.batch_shape, *self.event_shape])
        u = torch.rand(shape, device=self.device)
        v = self.Phi_alpha + (self.Phi_beta - self.Phi_alpha) * u
        return Phi_inv(v) * self.norm.scale + self.norm.loc
