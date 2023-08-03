from typing import Iterable

import torch
from torch import Tensor

from .distribution import Distribution
from .tensorlike import Tensorlike


class TensorTuple(tuple, Tensorlike):
    def __new__(cls, __iterable: Iterable, shape: torch.Size):
        return tuple.__new__(cls, __iterable)

    def __init__(self, __iterable: Iterable, shape: torch.Size):
        Tensorlike.__init__(self, shape)
        for idx, value in enumerate(self):
            self.register_field(str(idx), value)

    def _new(self, shape: torch.Size, fields: dict):
        values = [*self]
        for key, value in fields.items():
            values[int(key)] = value
        return TensorTuple(values, shape)


class TupleDist(Distribution, Tensorlike):
    def __init__(self, *dists: Distribution):
        dists = tuple(dists)
        Tensorlike.__init__(self, dists[0].batch_shape)
        self.register_field("dists", TensorTuple(dists, self.shape))

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
