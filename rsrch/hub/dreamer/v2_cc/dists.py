from numbers import Number

import torch
from torch import Tensor

from rsrch.distributions import *
from rsrch.types.tensorlike import Tensorlike


class TruncNormal(Normal):
    def __init__(
        self,
        loc: Tensor | Number,
        scale: Tensor | Number,
        low: Number,
        high: Number,
        clip=1e-6,
        mult=1.0,
        event_dims=0,
    ):
        super().__init__(loc, scale, event_dims)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()):
        sample = super().sample(sample_shape)
        if self._clip is not None:
            low, high = self._low + self._clip, self._high - self._clip
            sample = sample.clamp(low, high)
        else:
            sample = sample.clamp(self._low, self._high)
        if self._mult is not None:
            sample = sample * self._mult
        return sample

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        sample = super().rsample(sample_shape)
        zero_g = sample - sample.detach()
        if self._clip is not None:
            low, high = self._low + self._clip, self._high - self._clip
            sample = sample.clamp(low, high)
        else:
            sample = sample.clamp(self._low, self._high)
        if self._mult is not None:
            sample = sample * self._mult
        sample = sample + zero_g
        return sample

    def log_prob(self, value):
        logp = super().log_prob(value)
        # logp[(value <= self._low) | (value >= self._high)] = -torch.inf
        return logp
