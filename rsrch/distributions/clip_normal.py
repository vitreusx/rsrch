import math

import torch
from torch import Size, Tensor
from torch._C import Size

from rsrch import spaces
from rsrch.types import Tensorlike

from .distribution import Distribution
from .normal import Normal
from .utils import sum_rightmost


class ClipNormal(Distribution, Tensorlike):
    """Normal distribution clamped to a given space. log_prob and entropy
    functions are NOT mathematically correct."""

    def __init__(self, base: Normal, space: spaces.torch.Box):
        Tensorlike.__init__(self, base.shape)
        self.base = self.register("base", base)
        self.low = self.register("low", space.low, batched=False)
        self.high = self.register("high", space.high, batched=False)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.base.log_prob(value)

    def entropy(self) -> Tensor:
        return self.base.entropy()

    def rsample(self, sample_shape: Size = ()) -> Tensor:
        x = self.base.rsample(sample_shape)
        return x.clamp(self.low, self.high)
