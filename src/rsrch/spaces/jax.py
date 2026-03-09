from numbers import Number

import equinox as eqx
import jax
import jax.numpy as jnp


class Array(eqx.Module):
    shape: tuple[int, ...]
    dtype: jnp.dtype

    def __init__(self, shape: tuple[int, ...], *, dtype: jnp.dtype):
        self.shape = shape
        self.dtype = dtype


class Box(Array):
    low: jax.Array
    high: jax.Array

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        low: jax.Array | Number | None = None,
        high: jax.Array | Number | None = None,
        dtype: jnp.dtype | None = None,
    ):
        if dtype is None:
            low = jnp.asarray(low)
            dtype = low.dtype
        else:
            if low is None:
                if jnp.issubdtype(dtype, jnp.floating):
                    low = jnp.finfo(dtype).min
                else:
                    low = jnp.iinfo(dtype).min
            low = jnp.asarray(low, dtype=dtype)

        if high is None:
            if jnp.issubdtype(dtype, jnp.floating):
                high = jnp.finfo(dtype).max
            else:
                high = jnp.iinfo(dtype).max
        high = jnp.asarray(high, dtype=dtype)

        super().__init__(shape, dtype=dtype)
        self.low = jnp.broadcast_to(low, shape)
        self.high = jnp.broadcast_to(high, shape)

    @property
    def bounded_below(self):
        return ~jnp.isneginf(self.low)

    @property
    def bounded_above(self):
        return ~jnp.isposinf(self.high)

    @property
    def bounded(self):
        return self.bounded_below & self.bounded_above


class Discrete(Box):
    n: int

    def __init__(
        self,
        n: int,
        *,
        dtype: jnp.dtype = jnp.int32,
    ):
        if not jnp.issubdtype(dtype, jnp.integer):
            raise TypeError("Must provide integer dtype for a discrete space")
        super().__init__((), low=0, high=n, dtype=dtype)
        self.n = n


class Image(Box):
    channel_first: bool
    num_channels: int
    height: int
    width: int

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: jnp.dtype = jnp.float32,
        channel_first: bool = True,
    ):
        if dtype == jnp.uint8:
            low, high = 0, 255
        elif dtype == jnp.float32:
            low, high = 0.0, 1.0
        else:
            err = f"dtype must be uint8 or float32, is {dtype}"
            raise NotImplementedError(err)

        super().__init__(shape, low=low, high=high, dtype=dtype)
        self.channel_first = channel_first
        if channel_first:
            self.num_channels, self.height, self.width = shape
        else:
            self.height, self.width, self.num_channels = shape

    @property
    def size(self):
        return self.width, self.height
