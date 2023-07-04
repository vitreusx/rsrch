from typing import SupportsFloat

import numpy as np
import torch
import torch.distributions as D
from gymnasium.spaces import *
from torch import Tensor


class TensorBox(Space[Tensor]):
    def __init__(
        self,
        low: SupportsFloat | Tensor,
        high: SupportsFloat | Tensor,
        shape: torch.Size | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        seed: int | torch.Generator | None = None,
    ):
        super().__init__()

        self._shape: torch.Size
        if shape is not None:
            self._shape = torch.Size(shape)
        elif isinstance(low, Tensor):
            self._shape = low.shape
        elif isinstance(high, Tensor):
            self._shape = high.shape
        elif isinstance(low, SupportsFloat) and isinstance(high, SupportsFloat):
            self._shape = torch.Size([1])
        else:
            raise ValueError(f"Invalid shape {shape}.")

        if isinstance(low, Tensor):
            self.low = low
        elif isinstance(low, SupportsFloat):
            self.low = torch.full(self.shape, low, dtype=dtype, device=device)
        else:
            raise ValueError(
                f"low must be either Tensor or a float-like, is {type(low)}"
            )
        assert self.low.shape == self.shape

        if isinstance(high, Tensor):
            self.high = high
        elif isinstance(high, SupportsFloat):
            self.high = torch.full(self.shape, high, dtype=dtype, device=device)
        else:
            raise ValueError(
                f"high must be either Tensor or a float-like, is {type(high)}"
            )
        assert self.high.shape == self.shape

        self.dtype = dtype if dtype is not None else self.low.dtype
        assert self.low.dtype == self.dtype
        assert self.high.dtype == self.dtype

        self.device = torch.device(device) if device is not None else self.low.device
        assert self.low.device == self.device
        assert self.high.device == self.device

        if isinstance(seed, int):
            self.gen = torch.Generator(self.device)
            self.gen.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            self.gen = seed
            assert self.gen.device == self.device
        else:
            self.gen = torch.Generator(self.device)

        self.bounded_below = -torch.inf < self.low
        self.bounded_above = torch.inf > self.high

        self.sample_dt = self.dtype if self.dtype.is_floating_point else torch.float32
        self.eps = torch.finfo(self.sample_dt).tiny

        self._low_repr = self._short_repr(self.low)
        self._high_repr = self._short_repr(self.high)

    @staticmethod
    def from_numpy(box: Box, device: torch.device = None, dtype: torch.dtype = None):
        low = torch.from_numpy(box.low).to(device=device, dtype=dtype)
        high = torch.from_numpy(box.high).to(device=device, dtype=dtype)
        shape = torch.Size([*box.shape])
        dtype = low.dtype
        return TensorBox(low, high, shape, device, dtype)

    @staticmethod
    def _short_repr(x: Tensor) -> str:
        if x.size != 0 and torch.min(x).item() == torch.max(x).item():
            return str(torch.min(x).item())
        else:
            return str(x)

    def sample(self, mask=None) -> Tensor:
        sample = torch.empty(self.shape, device=self.device, dtype=self.sample_dt)

        # Unbounded -> normal distribution
        unbounded = ~self.bounded_below & ~self.bounded_above
        sample[unbounded] = torch.randn(
            unbounded[unbounded].shape,
            dtype=self.sample_dt,
            device=self.device,
            generator=self.gen,
        )

        # Upper-bounded: shifted negative exponential distribution
        # Note: -log(U)/\lambda ~ Exp(\lambda)
        upp_bounded = ~self.bounded_below & self.bounded_above
        U = torch.rand(
            upp_bounded[upp_bounded].shape,
            dtype=self.sample_dt,
            device=self.device,
            generator=self.gen,
        )
        sample[upp_bounded] = self.high[upp_bounded] + torch.log(U + self.eps)

        # Lower-bounded: shifted exponential distribution
        low_bounded = self.bounded_below & ~self.bounded_above
        U = torch.rand(
            low_bounded[low_bounded].shape,
            dtype=self.sample_dt,
            device=self.device,
            generator=self.gen,
        )
        sample[low_bounded] = self.low[low_bounded] - torch.log(U + self.eps)

        # Bounded: uniform
        bounded = self.bounded_below & self.bounded_above
        U = torch.rand(
            bounded[bounded].shape,
            dtype=self.sample_dt,
            device=self.device,
            generator=self.gen,
        )
        sample[bounded] = self.low[bounded] + U * (
            self.high[bounded] - self.low[bounded]
        )

        sample = sample.to(dtype=self.dtype)
        return sample

    def seed(self, seed: int | None = None) -> list[int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"TensorBox({self._low_repr}, {self._high_repr}, {tuple(self.shape)}, {self.dtype})"


class TensorDiscrete(Space[Tensor]):
    def __init__(
        self,
        n: int,
        device: torch.device | None = None,
        seed: int | torch.Generator | None = None,
        start: int = 0,
    ):
        super().__init__()
        self.n, self.start = n, start

        self.device = torch.device(device if device is not None else "cpu")

        if isinstance(seed, int):
            self.gen = torch.Generator(self.device)
            self.gen.manual_seed(seed)
        elif isinstance(seed, torch.Generator):
            self.gen = seed
            assert self.gen.device == self.device
        else:
            self.gen = torch.Generator(self.device)

        self.dtype = torch.int32
        self._shape = torch.Size([])

    @staticmethod
    def from_numpy(space: Discrete, device: torch.device = None):
        return TensorDiscrete(space.n, device=device)

    def sample(self, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            idxes = torch.where(mask)[0].to(dtype=self.dtype)
            if len(idxes) == 0:
                return torch.tensor(self.start, dtype=self.dtype, device=self.device)
            else:
                chosen = torch.randint(
                    high=len(idxes), size=[], device=self.device, generator=self.gen
                )
                return self.start + idxes[chosen]
        else:
            return torch.randint(
                low=self.start,
                high=self.start + self.n,
                size=[],
                device=self.device,
                dtype=self.dtype,
                generator=self.gen,
            )

    def __repr__(self) -> str:
        if self.start == 0:
            return f"TensorDiscrete({self.n})"
        else:
            return f"TensorDiscrete({self.n}, start={self.start})"
