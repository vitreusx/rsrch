from functools import partial
from numbers import Number

import numpy as np


class Array:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype | type | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def empty(self, shape: tuple[int, ...] = ()):
        return np.empty((*shape, *self.shape), dtype=self.dtype)

    def sample(
        self,
        shape: tuple[int, ...] = (),
        gen: np.random.Generator | None = None,
    ):
        raise NotImplementedError()

    def __repr__(self):
        return f"Space(shape={self.shape!r}, dtype={self.dtype})"


class Box(Array):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        low: np.ndarray | Number | None = None,
        high: np.ndarray | Number | None = None,
        dtype: np.dtype | None = None,
    ):
        if dtype is None:
            if low is None:
                dtype = np.float32
                low = -np.inf
            low = np.asarray(low, dtype)
            dtype = low.dtype
        else:
            dtype = np.dtype(dtype)
            if low is None:
                if np.issubdtype(dtype, np.floating):
                    low = -np.inf
                elif np.issubdtype(dtype, np.integer):
                    low = np.iinfo(dtype).min
                elif dtype == np.dtype(bool):
                    low = 0
            low = np.asarray(low, dtype)

        if high is None:
            if np.issubdtype(dtype, np.floating):
                high = np.inf
            elif np.issubdtype(dtype, np.integer):
                high = np.iinfo(dtype).max
            elif dtype == np.dtype(bool):
                high = 1
        high = np.asarray(high, dtype)

        super().__init__(shape, dtype=dtype)
        self.low = np.broadcast_to(low, shape)
        self.high = np.broadcast_to(high, shape)

    @property
    def bounded_below(self):
        return ~np.isneginf(self.low)

    @property
    def bounded_above(self):
        return ~np.isposinf(self.high)

    @property
    def bounded(self):
        return self.bounded_below & self.bounded_above

    def sample(
        self,
        sample_size: tuple[int, ...] = (),
        gen: np.random.Generator | None = None,
    ):
        shape = [*sample_size, *self.shape]
        if np.issubdtype(self.dtype, np.floating):
            if gen is None:
                u = np.random.rand(*shape).astype(self.dtype)
            else:
                u = gen.random(shape, self.dtype)
            u = np.where(self.bounded, u * (self.high - self.low), u)
            u = np.where(self.bounded_below, self.low + u, u)
        elif np.issubdtype(self.dtype, np.integer):
            if gen is None:
                u = np.random.randint(self.low, self.high, size=shape, dtype=self.dtype)
            else:
                u = gen.integers(self.low, self.high, size=shape, dtype=self.dtype)
        return u

    def __repr__(self):
        low_x = self.low.ravel()[0]
        low_r = low_x if np.all(self.low == low_x) else self.low
        high_x = self.high.ravel()[0]
        high_r = high_x if np.all(self.high == high_x) else self.high
        return f"Box({low_r!r}, {high_r!r}, {self.shape!r}, {self.dtype})"


class Discrete(Box):
    def __init__(
        self,
        n: int,
        *,
        dtype: np.dtype = np.int64,
    ):
        assert np.issubdtype(dtype, np.integer)
        super().__init__((), low=0, high=n, dtype=dtype)
        self.n = n

    def __repr__(self):
        return f"Discrete({self.n!r}, {self.dtype})"


class Image(Box):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: np.dtype = np.uint8,
        channel_last=True,
    ):
        if dtype == np.uint8:
            low, high = 0, 255
        elif dtype == np.float32:
            low, high = 0.0, 1.0
        super().__init__(shape, low=low, high=high, dtype=dtype)

        self.channel_last = channel_last
        if channel_last:
            self.height, self.width, self.num_channels = shape[-3:]
        else:
            self.num_channels, self.height, self.width = shape[-3:]
        self.size = self.width, self.height

    def __repr__(self):
        return f"Image({self.shape!r}, {self.dtype})"


class Dict(dict):
    def sample(
        self,
        shape: tuple[int, ...],
        gen: np.random.Generator | None = None,
    ):
        return {key: value.sample(shape, gen) for key, value in self.items()}
