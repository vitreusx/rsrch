from numbers import Number

import torch
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .utils import sum_rightmost


class Uniform(Distribution, Tensorlike):
    def __init__(self, low: Tensor | Number, high: Tensor | Number, event_dims=0):
        low = torch.as_tensor(low)
        high = torch.as_tensor(high, device=low.device)
        shape = torch.broadcast_shapes(low.shape, high.shape)

        pivot = len(shape) - event_dims
        batch_shape, event_shape = shape[:pivot], shape[pivot:]
        Tensorlike.__init__(self, batch_shape)

        self.register("low", low)
        self.register("high", high)
        self.event_shape = event_shape

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
        return self.low + (self.high - self.low) * torch.rand(
            size=shape,
            device=self.low.device,
            dtype=self.low.dtype,
        )

    def log_prob(self, value: Tensor):
        outside = (value < self.low) | (self.high <= value)
        logp = -(self.high - self.low).log()
        logp[outside] = torch.inf
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ent = (self.high - self.low).log()
        return sum_rightmost(ent, len(self.event_shape))
