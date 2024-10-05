import math
from numbers import Number

import torch
from torch import Tensor

from rsrch.nn.utils import pass_gradient
from rsrch.types.tensorlike import Tensorlike, defer_eval

from .affine import Affine
from .distribution import Distribution
from .normal import Normal
from .utils import sum_rightmost


def _normal_log_pdf(x: Tensor):
    return -0.5 * (math.log(2 * math.pi) + x.square())


def _normal_pdf(x: Tensor):
    return (-0.5 * x.square()).exp() / math.sqrt(2 * math.pi)


class TruncStdNormal(Distribution, Tensorlike):
    def __init__(self, low: Tensor, high: Tensor, event_dims: int = 0):
        batch_shape = low.shape[: len(low.shape) - event_dims]
        Tensorlike.__init__(self, batch_shape)
        self.event_dims = event_dims
        self.event_shape = low.shape[len(low.shape) - event_dims :]

        self.low = self.register("low", low)
        self.high = self.register("high", high)

        self.eps = torch.finfo(self.low.dtype).eps

    @defer_eval
    def low_pdf(self):
        return _normal_pdf(self.low)

    @defer_eval
    def high_pdf(self):
        return _normal_pdf(self.high)

    @defer_eval
    def low_cdf(self) -> Tensor:
        return torch.special.ndtr(self.low)

    @defer_eval
    def high_cdf(self) -> Tensor:
        return torch.special.ndtr(self.high)

    @defer_eval
    def Z(self):
        return (self.high_cdf - self.low_cdf).clamp_min(self.eps)

    @defer_eval
    def log_Z(self):
        return self.Z.log()

    @defer_eval
    def var_term(self):
        return (self.high * self.high_pdf - self.low * self.low_pdf) / self.Z

    @property
    def mean(self):
        return (self.low_pdf - self.high_pdf) / self.Z

    @property
    def var(self):
        return 1.0 - self.var_term - self.mean**2

    @property
    def mode(self):
        return self.mean.clamp(self.low, self.high)

    def entropy(self):
        ent = 0.5 * math.log(2.0 * math.pi * math.e) + self.log_Z - 0.5 * self.var_term
        return sum_rightmost(ent, self.event_dims)

    def log_prob(self, value: Tensor):
        logp = _normal_log_pdf(value) - self.log_Z
        return sum_rightmost(logp, self.event_dims)

    def rsample(self, sample_shape=()):
        shape = [*sample_shape, *self.batch_shape, *self.event_shape]
        p = torch.empty(shape, device=self.low.device)
        p.uniform_(self.eps, 1.0 - self.eps)
        p = self.low_cdf + p * self.Z
        return torch.special.ndtri(p.float())


class TruncNormal(Affine):
    def __init__(self, norm: Normal, low: Tensor | Number, high: Tensor | Number):
        std_low = (low - norm.loc) / norm.scale
        std_high = (high - norm.loc) / norm.scale
        std_dist = TruncStdNormal(std_low, std_high, len(norm.event_shape))
        super().__init__(std_dist, norm.loc, norm.scale, batched=True)
