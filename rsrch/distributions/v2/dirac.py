import torch
from torch import Tensor

from .utils import distribution


@distribution
class Dirac:
    value: Tensor
    event_shape: torch.Size

    def __init__(self, value: Tensor, event_dims: int):
        pivot = len(value.shape) - range(len(value.shape)).index(event_dims)
        self.__tc_init__(
            value=value,
            event_shape=value.shape[pivot:],
            batch_size=value.shape[:pivot],
        )

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def mean(self):
        return self.value

    @property
    def mode(self):
        return self.value

    @property
    def variance(self):
        return 0.0

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape=torch.Size()):
        return self.value.expand(*sample_shape, *self.value.shape)

    def log_prob(self, other):
        raise NotImplementedError

    def entropy(self):
        return 0.0
