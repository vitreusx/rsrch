from typing import Iterable

import torch
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike, TensorTuple

from .distribution import Distribution


class Tuple(Distribution, Tensorlike):
    def __init__(self, *dists: Distribution):
        dists = tuple(dists)
        Tensorlike.__init__(self, dists[0].batch_shape)
        self.register("dists", TensorTuple(dists, self.shape))

    @property
    def mean(self):
        return self._apply(lambda d: d.mean)

    @property
    def variance(self):
        return self._apply(lambda d: d.variance)

    @property
    def std(self):
        return self._apply(lambda d: d.std)

    @property
    def mode(self):
        return self._apply(lambda d: d.mode)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self._apply(lambda d: d.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return self._apply(lambda d: d.rsample(sample_shape))

    def log_prob(self, value: tuple) -> Tensor:
        return sum(d.log_prob(v) for d, v in zip(self.dists, value))

    def entropy(self) -> Tensor:
        return sum(d.entropy() for d in self.dists)

    def _apply(self, f):
        return TensorTuple((f(d) for d in self.dists), self.batch_shape)
