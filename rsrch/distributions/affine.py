import math
from functools import cached_property
from numbers import Number

import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .distribution import Distribution
from .utils import sum_rightmost


class Affine(Distribution, Tensorlike):
    def __init__(
        self,
        base: Distribution,
        loc: Number | Tensor,
        scale: Number | Tensor,
        batched: bool = False,
    ):
        Tensorlike.__init__(self, base.batch_shape)
        self.event_shape = base.event_shape
        self.event_dims = len(self.event_shape)

        self.base = self.register("base", base)

        if isinstance(loc, Tensor):
            self.loc = self.register("loc", loc, batched=batched)
        else:
            self.loc = loc

        if isinstance(scale, Tensor):
            self.scale = self.register("scale", scale, batched=batched)
        else:
            self.scale = scale

    @cached_property
    def log_scale(self):
        if isinstance(self.scale, Tensor):
            return sum_rightmost(self.scale.abs().log(), self.event_dims)
        else:
            return math.log(abs(self.scale))

    def log_prob(self, value: Tensor) -> Tensor:
        value = (value - self.loc) / self.scale
        return self.base.log_prob(value) - self.log_scale

    @property
    def mean(self):
        return self.loc + self.scale * self.base.mean

    @property
    def mode(self):
        return self.loc + self.scale * self.base.mode

    @property
    def var(self):
        return (self.scale**2) * self.base.var

    def entropy(self):
        return self.base.entropy() + self.log_scale

    def sample(self, sample_size: tuple[int, ...] = ()):
        return self.loc + self.scale * self.base.sample(sample_size)

    def rsample(self, sample_size: tuple[int, ...] = ()):
        return self.loc + self.scale * self.base.rsample(sample_size)
