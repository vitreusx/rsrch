from numbers import Number

import torch
from torch import Tensor

from .base import *


class TensorSpace:
    """Tensor equivalent of gym.Space. Is *not* actually a subclass of gym.Space,
    and cannot be used in actual envs. Used for input specification."""


class TensorBox(TensorSpace):
    def __init__(
        self,
        shape: torch.Size,
        low: Tensor | Number = None,
        high: Tensor | Number = None,
        dtype: torch.dtype = None,
    ):
        self.shape = shape
        self.low = low if low is not None else -torch.inf
        self.high = high if high is not None else +torch.inf
        self.dtype = dtype

    def sample(self):
        x = torch.rand(self.shape, dtype=self.dtype)
        x = x.clamp(self.low, self.high)
        return x


class TensorDiscrete(TensorSpace):
    def __init__(self, n: int, dtype: torch.dtype = None):
        self.n = n
        self.dtype = dtype
        self.shape = torch.Size([])

    def sample(self):
        return torch.randint(0, self.n, [])


class TensorImage(TensorSpace):
    def __init__(self, shape: torch.Size, dtype: torch.dtype = None):
        self.shape = shape
        self.num_channels, self.height, self.width = shape[-3:]
        self.dtype = dtype
        assert self.dtype in (torch.uint8, torch.float32)

    def sample(self):
        if self.dtype == torch.uint8:
            return torch.randint(0, 256, self.shape, dtype=self.dtype)
        else:
            return torch.rand(self.shape, dtype=self.dtype)
