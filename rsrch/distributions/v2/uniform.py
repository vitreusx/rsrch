from numbers import Number

import torch
from torch import Tensor

from .utils import _sum_rightmost, distribution


@distribution
class Uniform:
    low: Tensor
    high: Tensor
    event_shape: torch.Size

    def __init__(self, low: Tensor | Number, high: Tensor | Number, event_dims=0):
        low = torch.as_tensor(low)
        high = torch.as_tensor(high, device=low.device)
        shape = torch.broadcast_shapes(low.shape, high.shape)
        pivot = len(shape) - event_dims
        batch_shape, event_shape = shape[:pivot], shape[pivot:]
        self.__tc_init__(low, high, batch_size=batch_shape)
        self.event_shape = event_shape

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def mean(self):
        return (self.low + self.high) / 2

    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        shape = [*sample_shape, *self.batch_shape, *self.event_shape]
        shape = torch.Size(shape)
        return self.low + (self.high - self.low) * torch.rand(shape)

    def log_prob(self, value: Tensor):
        outside = (value < self.low) | (self.high <= value)
        logp = -(self.high - self.low).log()
        logp[outside] = torch.inf
        return _sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ent = (self.high - self.low).log()
        return _sum_rightmost(ent, len(self.event_shape))
