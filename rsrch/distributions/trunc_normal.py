import math
from functools import cached_property
from numbers import Number

import torch
from torch import Tensor

from rsrch.nn.utils import pass_gradient
from rsrch.types.tensorlike import Tensorlike

from .affine import Affine
from .distribution import Distribution
from .normal import Normal
from .utils import sum_rightmost

# def _log1mexp(x: Tensor):
#     """Compute log(1 - exp(x)) for x < 0."""
#     return torch.where(
#         -math.log(2) < x,
#         (-x.expm1()).log(),
#         (-x.exp()).log1p(),
#     )


# def _logsubexp(x: Tensor, y: Tensor):
#     """Compute log(exp(x) - exp(y)) for x >= y."""
#     return x + _log1mexp(y - x)


# def _subexp(x: Tensor, y: Tensor):
#     """Compute exp(x) - exp(y)."""
#     a, b = torch.maximum(x, y), torch.minimum(x, y)
#     return (x - y).sign() * _logsubexp(a, b).exp()


def _normal_log_pdf(x: Tensor):
    return -0.5 * (math.log(2 * math.pi) + x.square())


def _normal_pdf(x: Tensor):
    return (-0.5 * x.square()).exp() / math.sqrt(2 * math.pi)


# class TruncNormal(Distribution, Tensorlike):
#     eps = 1e-6

#     def __init__(self, dist: Normal, low: Tensor | Number, high: Tensor | Number):
#         Tensorlike.__init__(self, dist.shape)
#         self.event_shape = dist.event_shape
#         self.loc = self.register("loc", dist.loc)
#         self.scale = self.register("scale", dist.scale)

#         low = torch.as_tensor(low).type_as(self.loc)
#         self.low = self.register("low", low, batched=False)
#         high = torch.as_tensor(high).type_as(self.loc)
#         self.high = self.register("high", high, batched=False)

#     @cached_property
#     def std_low(self):
#         return (self.low - self.loc) / self.scale

#     @cached_property
#     def std_high(self):
#         return (self.high - self.loc) / self.scale

#     @cached_property
#     def log_Z(self):
#         x, y = self.std_low, self.std_high
#         is_y_pos = y >= 0
#         old_x = x
#         x = torch.where(is_y_pos, -y, x)
#         x_log_cdf: Tensor = torch.special.log_ndtr(x.float())
#         y = torch.where(is_y_pos, -old_x, y)
#         y_log_cdf: Tensor = torch.special.log_ndtr(y.float())
#         return y_log_cdf + _log1mexp(x_log_cdf - y_log_cdf)

#     @cached_property
#     def Z(self):
#         return self.log_Z.exp()

#     @cached_property
#     def std_low_log_pdf(self):
#         return _normal_log_pdf(self.std_low)

#     @cached_property
#     def std_low_pdf(self):
#         return _normal_pdf(self.std_low)

#     @cached_property
#     def std_high_log_pdf(self):
#         return _normal_log_pdf(self.std_high)

#     @cached_property
#     def std_high_pdf(self):
#         return _normal_pdf(self.std_high)

#     @cached_property
#     def mean_term(self):
#         phi_diff = _subexp(self.std_low_log_pdf, self.std_high_log_pdf)
#         return phi_diff / self.Z

#     @cached_property
#     def var_term(self):
#         return (
#             self.std_low * self.std_low_pdf - self.std_high * self.std_high_pdf
#         ) / self.Z

#     def log_prob(self, value: Tensor):
#         std_value = (value - self.loc) / self.scale
#         logp = -(
#             0.5 * std_value.square()
#             + 0.5 * math.log(2 * math.pi)
#             + self.scale.log()
#             + self.log_Z
#         )
#         logp = torch.where((value > self.high) | (value < self.low), -math.inf, logp)
#         return sum_rightmost(logp, len(self.event_shape))

#     def entropy(self):
#         ent = (
#             0.5 * (1.0 + math.log(2.0 * math.pi))
#             + self.scale.log()
#             + self.log_Z
#             + 0.5 * self.var_term
#         )
#         return sum_rightmost(ent, len(self.event_shape))

#     @property
#     def mean(self):
#         return self.loc + self.scale * self.mean_term

#     @property
#     def mode(self):
#         return self.loc.max(self.low).min(self.high)

#     @property
#     def var(self):
#         return self.scale.square() * (1.0 + self.var_term - self.mean_term.square())

#     def rsample(self, sample_shape=()):
#         std_low_cdf: Tensor = torch.special.ndtr(self.std_low)
#         std_high_cdf: Tensor = torch.special.ndtr(self.std_high)

#         shape = [*sample_shape, *self.batch_shape, *self.event_shape]
#         sample = torch.rand(shape, device=self.device)
#         sample = std_low_cdf + (std_high_cdf - std_low_cdf) * sample

#         sample: Tensor = torch.special.ndtri(sample.clamp(self.eps, 1.0 - self.eps))
#         sample = torch.nan_to_num(sample, neginf=0.0, posinf=0.0)

#         sample = self.loc + self.scale * sample
#         sample = pass_gradient(sample.clamp(self.low, self.high), sample)
#         return sample


class TruncStdNormal(Distribution, Tensorlike):
    def __init__(self, low: Tensor, high: Tensor, event_dims: int = 0):
        batch_shape = low.shape[: len(low.shape) - event_dims]
        Tensorlike.__init__(self, batch_shape)
        self.event_dims = event_dims
        self.event_shape = low.shape[len(low.shape) - event_dims :]

        self.low = self.register("low", low)
        self.high = self.register("high", high)

        self.eps = torch.finfo(self.low.dtype).eps

    @cached_property
    def low_pdf(self):
        return _normal_pdf(self.low)

    @cached_property
    def high_pdf(self):
        return _normal_pdf(self.high)

    @cached_property
    def low_cdf(self) -> Tensor:
        return torch.special.ndtr(self.low)

    @cached_property
    def high_cdf(self) -> Tensor:
        return torch.special.ndtr(self.high)

    @cached_property
    def Z(self):
        return (self.high_cdf - self.low_cdf).clamp_min(self.eps)

    @cached_property
    def log_Z(self):
        return self.Z.log()

    @cached_property
    def var_term(self):
        return (self.high * self.high_pdf - self.low * self.low_pdf) / self.Z

    @property
    def mean(self):
        return (self.low_pdf - self.high_pdf) / self.Z

    @property
    def variance(self):
        return 1.0 - self.var_term - self.mean**2

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
