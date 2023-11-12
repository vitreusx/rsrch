from numbers import Number

import numpy as np


class Space:
    ...


class Box:
    def __init__(
        self,
        shape: tuple[int, ...],
        low: np.ndarray | Number,
        high: np.ndarray | Number,
        dtype: np.dtype,
        seed: np.random.Generator | None = None,
    ):
        ...


class Discrete:
    def __init__(
        self,
        n: int,
        dtype: np.dtype = np.int64,
        seed: np.random.Generator | None = None,
    ):
        ...


class Image(Box):
    def __init__(
        self,
        size: tuple[int, int],
        num_channels: int,
        channel_last=True,
        dtype: np.dtype = np.uint8,
        seed: np.random.Generator | None = None,
    ):
        self.size = size
        self.num_channels = num_channels
        w, h = size
