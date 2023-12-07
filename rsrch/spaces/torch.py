from numbers import Number

import torch
from torch import Tensor


class Space:
    _TORCH_FUNCTIONS = {}

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
        if isinstance(seed, torch.Generator):
            self._gen = seed
        else:
            self._seed, self._gen = seed, None

    @property
    def gen(self):
        if self._gen is None:
            self._gen = torch.Generator(self.device)
            self._gen.manual_seed(self._seed)
        return self._gen

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        args_ = []
        for arg in args:
            if isinstance(arg, Space):
                args_.append(torch.empty(arg.shape, arg.dtype, arg.device))
            else:
                args_.append(arg)

        kwargs = {} if kwargs is None else kwargs
        res: Tensor = func(*args, **kwargs)
        return Space(res.shape, res.dtype, res.device)

    def __repr__(self):
        return f"Space({self.shape!r}, {self.dtype}, {self.device})"


class Box(Space):
    def __init__(
        self,
        shape: tuple[int, ...],
        low: Tensor | Number,
        high: Tensor | Number,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        seed: torch.Generator | int | None = None,
    ):
        if dtype is None:
            low = torch.as_tensor(low, device=device)
            dtype = low.dtype
        else:
            low = torch.as_tensor(low, dtype=dtype, device=device)
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
        if isinstance(self.dtype, torch.float):
            u = torch.rand(
                shape, dtype=self.dtype, device=self.device, generator=self.gen
            )
            u = torch.where(self.bounded, u * (self.high - self.low), u)
            u = torch.where(self.bounded_below, self.low + u, u)
        else:
            u = torch.randint(
                self.low,
                self.high,
                shape,
                dtype=self.dtype,
                device=self.device,
                generator=self.gen,
            )

        return u

    def __repr__(self):
        low_x = self.low.ravel()[0].item()
        low_r = low_x if torch.all(self.low == low_x) else self.low
        high_x = self.high.ravel()[0].item()
        high_r = high_x if torch.all(self.high == high_x) else self.high
        return (
            f"Box({low_r!r}, {high_r!r}, {self.shape!r}, {self.dtype}, {self.device})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        low_args, high_args = [], []
        for arg in args:
            if isinstance(arg, Box):
                low_args.append(arg.low)
                high_args.append(arg.high)
            else:
                low_args.append(arg)
                high_args.append(arg)

        kwargs = {} if kwargs is None else kwargs
        low_res: Tensor = func(*low_args, **kwargs)
        high_res: Tensor = func(*high_args, **kwargs)
        low = torch.minimum(low_res, high_res)
        high = torch.maximum(low_res, high_res)
        return Box(low.shape, low, high, low.dtype, low.device)


class Discrete(Space):
    def __init__(
        self,
        n: int,
        dtype: torch.dtype = torch.int64,
        device: torch.device | None = None,
        seed: torch.Generator | None = None,
    ):
        assert isinstance(dtype, torch.int)
        super().__init__((), dtype, device, seed)
        self.n = n

    def sample(self, sample_size: tuple[int, ...] = ()):
        shape = [*sample_size, *self.shape]
        return torch.randint(
            0, self.n, shape, dtype=self.dtype, device=self.device, generator=self.gen
        )

    def __repr__(self):
        return f"Discrete({self.n!r}, {self.dtype}, {self.device})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented


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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented
