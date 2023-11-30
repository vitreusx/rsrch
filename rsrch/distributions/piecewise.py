from numbers import Number

import numpy as np
import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .categorical import Categorical
from .distribution import Distribution
from .utils import sum_rightmost


class Piecewise(Tensorlike, Distribution):
    def __init__(
        self,
        dist: Categorical,
        low: Number | Tensor,
        high: Number | Tensor,
        event_dims: int,
    ):
        Tensorlike.__init__(self, dist.shape)
        self.dist = self.register("dist", dist)
        self.k = self.dist._num_events
        self.low, self.high = low, high
        self.step = (high - low) / self.k
        self.event_dims = event_dims

    @property
    def mean(self):
        centers = torch.arange(self.k, dtype=float, device=self.device) + 0.5
        return (centers * self.dist.probs).mean(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        bucket = ((value - self.low) / self.step).floor().long()
        logp = self.dist.log_prob(bucket) - self.step.log()
        return sum_rightmost(logp, self.event_dims)

    def entropy(self) -> Tensor:
        ent = self.dist.entropy() + torch.log(self.high - self.low) - np.log(self.k)
        return sum_rightmost(ent, self.event_dims)

    def rsample(self, sample_size: tuple[int, ...] = tuple()):
        bucket = self.dist.sample(sample_size).float()
        bucket = bucket + torch.rand_like(bucket)
        return self.low + self.step * bucket

    def sample(self, sample_size: tuple[int, ...] = tuple()):
        with torch.no_grad():
            return self.rsample(sample_size)
