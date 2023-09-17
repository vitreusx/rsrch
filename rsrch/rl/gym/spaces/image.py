from copy import deepcopy

import numpy as np

from ..vector.utils import batch_space
from .base import *


class Image(Box):
    """Box space interpreted as an Image. Behaves exactly like a Box; mostly used
    for observation type inference."""

    def __init__(
        self,
        shape: tuple[int, ...],
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
        self._normalized = normalized
        self._channels_last = channels_last

    @property
    def num_channels(self):
        if self._channels_last:
            return self.shape[-1]
        else:
            return self.shape[-3]

    def __repr__(self):
        return "Image" + super().__repr__()[len("Box") :]


@batch_space.register(Image)
def _(space: Image, n: int = 1):
    return Image(
        shape=[n, *space.shape],
        normalized=space._normalized,
        seed=deepcopy(space._np_random),
    )
