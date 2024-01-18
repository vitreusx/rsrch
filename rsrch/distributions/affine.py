from numbers import Number

import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .distribution import Distribution


class Affine(Distribution, Tensorlike):
    def __init__(
        self,
        base: Distribution,
        loc: Number | Tensor,
        scale: Number | Tensor,
    ):
        Tensorlike.__init__(self, base.batch_shape)
        self.event_shape = base.event_shape
        self.base = self.register("base", base)
        loc = torch.as_tensor(loc)
        self.loc = self.register("loc", loc, batched=False)
        scale = torch.as_tensor(scale)
        self.scale = self.register("scale", scale, batched=False)
        self.log_scale = self.register(
            "log_scale", scale.abs().log().sum(), batched=False
        )

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
        return self.scale.square() * self.base.var

    def entropy(self):
        return self.base.entropy() + self.log_scale

    def sample(self, sample_size: tuple[int, ...] = ()):
        return self.loc + self.scale * self.base.sample(sample_size)

    def rsample(self, sample_size: tuple[int, ...] = ()):
        return self.loc + self.scale * self.base.rsample(sample_size)
