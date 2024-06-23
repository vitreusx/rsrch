import math
from functools import cached_property
from numbers import Number

import numpy as np
import torch
from torch import Tensor, nn

from rsrch.types import Tensorlike

from .distribution import Distribution
from .normal import Normal
from .utils import sum_rightmost


def norm_pmf(x: Tensor):
    return (-0.5 * x.square()).exp() / math.sqrt(2 * math.pi)


def norm_cdf(x: Tensor):
    return 0.5 * (1.0 + (x / math.sqrt(2)).erf())


def norm_qf(x: Tensor, eps=1e-7):
    z = (2.0 * x - 1.0).clamp(-1.0 + eps, 1.0 - eps)
    return np.sqrt(2.0) * torch.erfinv(z)


class TruncNormal(Distribution, Tensorlike):
    def __init__(self, norm: Normal, low: Tensor, high: Tensor):
        Tensorlike.__init__(self, norm.batch_shape)
        self.event_shape = norm.event_shape

        self.norm = self.register("norm", norm)

        shape = [*self.batch_shape, *self.event_shape]
        self.low = self.register("low", low.expand(shape))
        self.high = self.register("high", high.expand(shape))

        self.low_std = self.register(
            "low_std",
            (self.low - self.norm.loc) / self.norm.scale,
        )
        self.low_pmf = self.register("low_pmf", norm_pmf(self.low_std))
        self.low_cdf = self.register("low_cdf", norm_cdf(self.low_std))

        self.high_std = self.register(
            "high_std",
            (self.high - self.norm.loc) / self.norm.scale,
        )
        self.high_pmf = self.register("high_pmf", norm_pmf(self.high_std))
        self.high_cdf = self.register("high_cdf", norm_cdf(self.high_std))

        self.pmf_z = self.register("pmf_z", self.high_cdf - self.low_cdf)

    def entropy(self):
        norm_ent = 0.5 + 0.5 * math.log(2.0 * math.pi) + self.norm.scale.log()
        trunc_ent = self.pmf_z.log() + (
            0.5
            * (self.low_std * self.low_pmf - self.high_std * self.high_pmf)
            / self.pmf_z
        )
        ent = norm_ent + trunc_ent
        return sum_rightmost(ent, len(self.event_shape))

    def log_prob(self, value: Tensor):
        norm_logp = (
            -((value - self.norm.loc) ** 2) / (2 * self.norm.var)
            - self.norm.scale.log()
            - 0.5 * math.log(2 * math.pi)
        )
        trunc_logp = -self.pmf_z.log()
        logp = norm_logp + trunc_logp
        return sum_rightmost(logp, len(self.event_shape))

    def sample(self, sample_shape=()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=()):
        shape = torch.Size([*sample_shape, *self.batch_shape, *self.event_shape])
        u = torch.rand(shape, device=self.device)
        v = self.low_cdf + self.pmf_z * u
        return norm_qf(v) * self.norm.scale + self.norm.loc
