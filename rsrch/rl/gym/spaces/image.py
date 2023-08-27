from .base import Box
import typing
from .tensor import TensorBox
import torch
import numpy as np


__all__ = ["Image", "TensorImage"]


class Image(Box):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        normalized=False,
        channels_last=True,
        seed=None,
    ):
        if normalized:
            low, high = 0.0, 1.0
            dtype = np.float32
        else:
            low, high = 0, 255
            dtype = np.uint8

        super().__init__(low, high, shape, dtype, seed)
        self._channels_last = channels_last
        self._normalized = normalized

    @property
    def num_channels(self):
        if self._channels_last:
            return self.shape[-1]
        else:
            return self.shape[-3]

    def __repr__(self):
        return "Image" + super().__repr__()[len("Box") :]


class TensorImage(TensorBox):
    def __init__(
        self,
        shape: torch.Size,
        normalized=False,
        device=None,
        seed=None,
    ):
        if normalized:
            low, high = 0.0, 1.0
            dtype = torch.float32
        else:
            low, high = 0, 255
            dtype = torch.uint8

        super().__init__(low, high, shape, device, dtype, seed)
        self._normalized = normalized

    @property
    def num_channels(self):
        if self._channels_last:
            return self.shape[-1]
        else:
            return self.shape[-3]

    def __repr__(self):
        return "TensorImage" + super().__repr__()[len("TensorBox") :]
