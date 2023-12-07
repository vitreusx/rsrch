from numbers import Number

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class Space(NDArrayOperatorsMixin):
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype | type | None = None,
        seed: np.random.Generator | int | None = None,
    ):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        if isinstance(seed, np.random.Generator):
            self._gen = seed
        else:
            self._seed, self._gen = seed, None

    @property
    def gen(self) -> np.random.Generator:
        if self._gen is None:
            self._gen = np.random.default_rng(self._seed)
        return self._gen

    def __repr__(self):
        return f"Space({self.shape!r}, {self.dtype})"

    def __array__(self, dtype=None):
        return np.empty(self.shape, dtype or self.dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == "__call__":
            args_ = [x.__array__() if isinstance(x, Space) else x for x in args]
            res: np.ndarray = ufunc(*args_, **kwargs)
            return self.__class__(res.shape, res.dtype)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        args_ = [x.__array__() if isinstance(x, Space) else x for x in args]
        res: np.ndarray = func(*args_, **kwargs)
        return self.__class__(res.shape, res.dtype)


class Box(Space):
    def __init__(
        self,
        shape: tuple[int, ...],
        low: np.ndarray | Number,
        high: np.ndarray | Number,
        dtype: np.dtype | None = None,
        seed: np.random.Generator | None = None,
    ):
        if dtype is None:
            low = np.asarray(low)
            dtype = low.dtype
        else:
            low = np.asarray(low, dtype)
        high = np.asarray(high, dtype)

        super().__init__(shape, dtype, seed)
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

    def sample(self, sample_size: tuple[int, ...] = ()):
        shape = [*sample_size, *self.shape]
        if np.issubdtype(self.dtype, np.floating):
            u = self.gen.random(shape, self.dtype)
            u = np.where(self.bounded, u * (self.high - self.low), u)
            u = np.where(self.bounded_below, self.low + u, u)
        elif np.issubdtype(self.dtype, np.integer):
            u = self.gen.integers(self.low, self.high, shape, self.dtype)
        return u

    def __repr__(self):
        low_x = self.low.ravel()[0]
        low_r = low_x if np.all(self.low == low_x) else self.low
        high_x = self.high.ravel()[0]
        high_r = high_x if np.all(self.high == high_x) else self.high
        return f"Box({low_r!r}, {high_r!r}, {self.shape!r}, {self.dtype})"

    def __array__(self, dtype=None):
        return self.low.copy()

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == "__call__":
            low_args, high_args = [], []
            for x in args:
                if isinstance(x, Box):
                    low_args.append(x.low)
                    high_args.append(x.high)
                else:
                    low_args.append(x)
                    high_args.append(x)

            low_res = ufunc(*low_args, **kwargs)
            high_res = ufunc(*high_args, **kwargs)
            low = np.minimum(low_res, high_res)
            high = np.maximum(low_res, high_res)

            return self.__class__(low.shape, low, high, low.dtype)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        low_args, high_args = [], []
        for x in args:
            if isinstance(x, Box):
                low_args.append(x.low)
                high_args.append(x.high)
            else:
                low_args.append(x)
                high_args.append(x)

        low_res = func(*low_args, **kwargs)
        high_res = func(*high_args, **kwargs)
        low = np.minimum(low_res, high_res)
        high = np.maximum(low_res, high_res)

        return self.__class__(low.shape, low, high, low.dtype)


class Discrete(Box):
    def __init__(
        self,
        n: int,
        dtype: np.dtype = np.int64,
        seed: np.random.Generator | None = None,
    ):
        assert np.issubdtype(dtype, np.integer)
        super().__init__((), 0, n, dtype, seed)
        self.n = n

    def sample(self, sample_size: tuple[int, ...] = ()):
        shape = [*sample_size, *self.shape]
        return self.gen.integers(0, self.n, shape, self.dtype)

    def __repr__(self):
        return f"Discrete({self.n!r}, {self.dtype})"

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented


class Image(Box):
    def __init__(
        self,
        shape: tuple[int, int, int],
        channel_last=True,
        dtype: np.dtype = np.uint8,
        seed: np.random.Generator | None = None,
    ):
        if dtype == np.uint8:
            low, high = 0, 255
        elif dtype == np.float32:
            low, high = 0.0, 1.0
        super().__init__(shape, low, high, dtype, seed)

        self.channel_last = channel_last
        if channel_last:
            self.height, self.width, self.num_channels = shape
        else:
            self.num_channels, self.height, self.width = shape
        self.size = self.width, self.height

    def __repr__(self):
        return f"Image({self.shape!r}, {self.dtype})"

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented
