from numbers import Number

import torch
from torch import Tensor

from . import np as spaces_np


class Space:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device or "cpu")

    def empty(self, shape: tuple[int, ...] = ()):
        return torch.empty([*shape, *self.shape], dtype=self.dtype, device=self.device)

    def sample(
        self,
        shape: tuple[int, ...] = (),
        gen: torch.Generator | None = None,
    ):
        raise NotImplementedError()

    def __repr__(self):
        return f"Space({self.shape!r}, {self.dtype}, {self.device})"


def to_tensor(x, dtype=None, device=None):
    if isinstance(x, Tensor):
        return x.detach().clone().to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)


class Box(Space):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        low: Tensor | Number | None = None,
        high: Tensor | Number | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        if dtype is None:
            low = to_tensor(low, device=device)
            dtype = low.dtype
        else:
            if low is None:
                if dtype.is_floating_point:
                    low = torch.finfo(dtype).min
                else:
                    low = torch.iinfo(dtype).min
            low = to_tensor(low, dtype=dtype, device=device)

        if high is None:
            if dtype.is_floating_point:
                high = torch.finfo(dtype).max
            else:
                high = torch.iinfo(dtype).max
        high = to_tensor(high, dtype=dtype, device=device)

        super().__init__(shape, dtype=dtype, device=device)
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

    def sample(
        self,
        sample_size: tuple[int, ...] = (),
        shape: tuple[int, ...] = (),
        gen: torch.Generator | None = None,
    ):
        shape = [*sample_size, *self.shape]
        if self.dtype.is_floating_point:
            u = torch.rand(
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=gen,
            )
            u = torch.where(self.bounded, u * (self.high - self.low), u)
            u = torch.where(self.bounded_below, self.low + u, u)
        else:
            u = torch.rand(
                shape,
                dtype=torch.float32,
                device=self.device,
                generator=gen,
            )
            u = (self.high - self.low).float() * u + self.low
            u = u.clamp(self.low, self.high)
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
        *,
        dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
    ):
        assert not dtype.is_floating_point
        super().__init__((), dtype=dtype, device=device)
        self.n = n

    def sample(
        self,
        sample_size: tuple[int, ...] = (),
        gen: torch.Generator | None = None,
    ):
        return torch.randint(
            0,
            self.n,
            sample_size,
            dtype=self.dtype,
            device=self.device,
            generator=gen,
        )

    def __repr__(self):
        return f"Discrete({self.n!r}, {self.dtype}, {self.device})"


class TokenSeq(Space):
    def __init__(
        self,
        num_tokens: int,
        vocab_size: int,
        *,
        dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
    ):
        super().__init__((num_tokens,), dtype=dtype, device=device)
        self.num_tokens, self.vocab_size = num_tokens, vocab_size

    def sample(
        self,
        sample_size: tuple[int, ...] = (),
        gen: torch.Generator | None = None,
    ):
        return torch.randint(
            0,
            self.vocab_size,
            (*sample_size, self.num_tokens),
            dtype=self.dtype,
            device=self.device,
            generator=gen,
        )

    def __repr__(self):
        attrs = ["num_tokens", "vocab_size", "dtype", "device"]
        arg = ",".join(f"{attr}={getattr(self, attr)!r}" for attr in attrs)
        return f"{self.__class__.__name__}({arg})"


class Image(Box):
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
        low: Tensor | Number | None = None,
        high: Tensor | Number | None = None,
        device: torch.device | None = None,
        channel_first: bool = True,
    ):
        if low is None:
            if dtype == torch.uint8:
                low, high = 0, 255
            elif dtype == torch.float32:
                low, high = 0.0, 1.0
            else:
                err = f"dtype must be uint8 or float32, is {dtype}"
                raise NotImplementedError(err)

        super().__init__(shape, low=low, high=high, dtype=dtype, device=device)
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
            low=to_tensor(space.low, device=device),
            high=to_tensor(space.high, device=device),
        )
    elif type(space) == spaces_np.Discrete:
        return Discrete(space.n, device=device)
    elif type(space) == spaces_np.Image:
        return Image(
            shape=space.shape,
            device=device,
            channel_first=not space.channel_last,
        )
