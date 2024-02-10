from numbers import Number

import torch
from torch import Tensor

from . import np as spaces_np


class Space:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        seed: torch.Generator | int | None = None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device or "cpu")
        self._seed = seed
        self._gen = seed if isinstance(seed, torch.Generator) else None

    @property
    def gen(self):
        if self._gen is None and self._seed is not None:
            self._gen = torch.Generator(self.device)
            self._gen.manual_seed(self._seed)
        return self._gen

    def empty(self, shape: tuple[int, ...] = ()):
        return torch.empty([*shape, *self.shape], dtype=self.dtype, device=self.device)

    def __repr__(self):
        return f"Space({self.shape!r}, {self.dtype}, {self.device})"


class Box(Space):
    def __init__(
        self,
        shape: tuple[int, ...],
        low: Tensor | Number | None = None,
        high: Tensor | Number | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        seed: torch.Generator | int | None = None,
    ):
        if dtype is None:
            low = torch.as_tensor(low, device=device)
            dtype = low.dtype
        else:
            if low is None:
                if dtype.is_floating_point:
                    low = torch.finfo(dtype).min
                else:
                    low = torch.iinfo(dtype).min
            low = torch.as_tensor(low, dtype=dtype, device=device)

        if high is None:
            if dtype.is_floating_point:
                high = torch.finfo(dtype).max
            else:
                high = torch.iinfo(dtype).max
        high = torch.as_tensor(high, dtype=dtype, device=device)

        super().__init__(shape, dtype, device, seed)
        self.low = low.expand(shape)
        self.high = high.expand(shape)

    @property
    def bounded_below(self):
        return ~torch.isneginf(self.low)

    @property
    def bounded_above(self):
        return ~torch.isposinf(self.high)

    @property
    def bounded(self):
        return self.bounded_below & self.bounded_above

    def sample(self, sample_size: tuple[int, ...] = ()):
        shape = [*sample_size, *self.shape]
        if self.dtype.is_floating_point:
            u = torch.rand(
                shape, dtype=self.dtype, device=self.device, generator=self.gen
            )
            u = torch.where(self.bounded, u * (self.high - self.low), u)
            u = torch.where(self.bounded_below, self.low + u, u)
        else:
            u = torch.rand(
                shape, dtype=torch.float32, device=self.device, generator=self.gen
            )
            u = (self.high - self.low).float() * u + self.low
            u = u.clamp(self.low, self.high - 1e-8)
            u = u.to(self.dtype)

        return u

    def __repr__(self):
        low_x = self.low.ravel()[0].item()
        low_r = low_x if torch.all(self.low == low_x) else self.low
        high_x = self.high.ravel()[0].item()
        high_r = high_x if torch.all(self.high == high_x) else self.high
        return (
            f"Box({low_r!r}, {high_r!r}, {self.shape!r}, {self.dtype}, {self.device})"
        )

    def __getitem__(self, index):
        low, high = self.low[index], self.high[index]
        return Box(
            shape=low.shape,
            low=low,
            high=high,
            dtype=self.dtype,
            device=self.device,
        )


class Discrete(Space):
    def __init__(
        self,
        n: int,
        dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
        seed: torch.Generator | None = None,
    ):
        assert not dtype.is_floating_point
        super().__init__((), dtype, device, seed)
        self.n = n

    def sample(self, sample_size: tuple[int, ...] = ()):
        shape = [*sample_size, *self.shape]
        return torch.randint(
            0, self.n, shape, dtype=self.dtype, device=self.device, generator=self.gen
        )

    def __repr__(self):
        return f"Discrete({self.n!r}, {self.dtype}, {self.device})"


class Image(Box):
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.uint8,
        device: torch.device | None = None,
        channel_first: bool = True,
        seed: torch.Generator | int | None = None,
    ):
        if dtype == torch.uint8:
            low, high = 0, 255
        elif dtype == torch.float32:
            low, high = 0.0, 1.0
        else:
            err = f"dtype must be uint8 or float32, is {dtype}"
            raise NotImplementedError(err)

        super().__init__(shape, low, high, dtype, device, seed)
        self.channel_first = channel_first
        if channel_first:
            self.num_channels, self.height, self.width = shape
        else:
            self.height, self.width, self.num_channels = shape
        self.size = self.width, self.height

    def __repr__(self):
        return f"Image({self.shape!r}, {self.dtype}, {self.device})"

    def __getitem__(self, index):
        shape = self.low[index].shape
        assert len(shape) >= 3
        return Image(
            shape=shape,
            dtype=self.dtype,
            device=self.device,
            channel_first=self.channel_first,
        )


def as_tensor(space: spaces_np.Space, device: torch.device | None = None):
    if type(space) == spaces_np.Box:
        return Box(
            shape=space.shape,
            low=torch.as_tensor(space.low, device=device),
            high=torch.as_tensor(space.high, device=device),
        )
    elif type(space) == spaces_np.Discrete:
        return Discrete(space.n, device=device)
    elif type(space) == spaces_np.Image:
        return Image(
            shape=space.shape,
            device=device,
            channel_first=not space.channel_last,
        )
