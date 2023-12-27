import math

import torch
from torch import Size, Tensor
from torch._C import Size

from rsrch.types import Tensorlike

from .distribution import Distribution
from .normal import Normal
from .utils import sum_rightmost


class ClipNormal(Distribution, Tensorlike):
    def __init__(self, base: Normal, low: Tensor, high: Tensor):
        Tensorlike.__init__(self, base.shape)
        self.event_shape = base.event_shape
        self.base = self.register("base", base)
        self.low = self.register("low", low, batched=False)
        self._low_logp = self.register(
            "_low_logp",
            self.base.log_cdf(low.expand_as(base.loc)),
        )
        self.high = self.register("high", high, batched=False)
        self._high_logp = self.register(
            "_high_logp",
            self.base.log_sf(high.expand_as(base.loc)),
        )

    def log_prob(self, value: Tensor) -> Tensor:
        reg_logp = (
            -((value - self.base.loc) ** 2) / (2 * self.base.var)
            - self.base.scale.log()
            - 0.5 * math.log(2 * math.pi)
        )
        logp = torch.where(
            value <= self.low,
            self._low_logp,
            torch.where(
                value < self.high,
                reg_logp,
                self._high_logp,
            ),
        )
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self) -> Tensor:
        # NOTE: Mathematically incorrect
        return self.base.entropy()

    def rsample(self, sample_shape: Size = ()) -> Tensor:
        x = self.base.rsample(sample_shape)
        return x.clamp(self.low, self.high)
