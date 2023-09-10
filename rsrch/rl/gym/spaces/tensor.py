import typing
from typing import SupportsFloat

import torch
from .base import Space, Tuple
from torch import Tensor

__all__ = ["TensorSpace", "TensorBox", "TensorDiscrete"]


class TensorSpace(Space[Tensor]):
    def __init__(
        self,
        shape: typing.Sequence[int] | torch.Size,
        dtype: torch.dtype,
        device: torch.device | None = None,
        seed: int | torch.Generator | None = None,
    ):
        super().__init__()
        self._shape = torch.Size(shape)
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        if isinstance(seed, (int, type(None))):
            self._seed = seed
            self._gen = None
        elif isinstance(seed, torch.Generator):
            self._seed = None
            self._gen = seed

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def gen(self) -> torch.Generator:
        if self._gen is None:
            self._gen = torch.Generator(self.device)
            if self._seed is not None:
                self._gen = self._gen.manual_seed(self._seed)
        return self._gen

    @gen.setter
    def _(self, value):
        self._gen = value

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is None:
            seed = self.gen.seed()
        self._gen = self.gen.manual_seed(seed)
        return [seed]

    def __getstate__(self):
        state = dict(self.__dict__)
        if state["_gen"] is not None:
            # torch.Generator objects are not pickleable - save state
            state["_gen_state"] = state["_gen"].get_state()
            del state["_gen"]
        return state

    def __setstate__(self, state):
        state = dict(state)
        if "_gen_state" in state:
            # Recover torch.Generator from saved state
            device = state["device"]
            gen = torch.Generator(device).set_state(state["_gen_state"])
            state["_gen"] = gen
        self.__dict__.update(state)


class TensorBox(TensorSpace):
    def __init__(
        self,
        low: SupportsFloat | Tensor | None = None,
        high: SupportsFloat | Tensor | None = None,
        shape: torch.Size | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        seed: int | torch.Generator | None = None,
    ):
        if low is None:
            low = -torch.inf
        if high is None:
            high = torch.inf

        if shape is not None:
            shape = torch.Size(shape)
        elif isinstance(low, Tensor):
            shape = low.shape
        elif isinstance(high, Tensor):
            shape = high.shape
        elif isinstance(low, SupportsFloat) and isinstance(high, SupportsFloat):
            shape = torch.Size([1])
        else:
            raise ValueError(f"Invalid shape {shape}.")

        if isinstance(low, Tensor):
            ...
        elif isinstance(low, SupportsFloat):
            low = torch.full(shape, low, dtype=dtype, device=device)
        else:
            raise ValueError(
                f"low must be either Tensor or a float-like, is {type(low)}"
            )
        assert low.shape == shape

        if isinstance(high, Tensor):
            ...
        elif isinstance(high, SupportsFloat):
            high = torch.full(shape, high, dtype=dtype, device=device)
        else:
            raise ValueError(
                f"high must be either Tensor or a float-like, is {type(high)}"
            )
        assert high.shape == shape

        dtype = dtype if dtype is not None else low.dtype
        assert low.dtype == dtype
        assert high.dtype == dtype

        device = torch.device(device) if device is not None else low.device
        assert low.device.type == device.type
        assert high.device.type == device.type

        super().__init__(shape, dtype, device, seed)
        self.low = low
        self.high = high

        self.bounded_below = -torch.inf < self.low
        self.bounded_above = torch.inf > self.high

        self.sample_dt = dtype if self.dtype.is_floating_point else torch.float32
        self.eps = torch.finfo(self.sample_dt).tiny

        self._low_repr = self._short_repr(self.low)
        self._high_repr = self._short_repr(self.high)

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

    def __eq__(self, other):
        if not isinstance(other, TensorBox):
            return False
        return (
            torch.allclose(self.low, other.low)
            and torch.allclose(self.high, other.high)
            and self.dtype == other.dtype
            and self.device == other.device
        )

    def __repr__(self) -> str:
        return f"TensorBox({self._low_repr}, {self._high_repr}, {tuple(self.shape)}, {self.dtype})"


class TensorDiscrete(TensorSpace):
    def __init__(
        self,
        n: int,
        device: torch.device | None = None,
        seed: int | torch.Generator | None = None,
        start: int = 0,
    ):
        super().__init__(shape=[], dtype=torch.int32, device=device, seed=seed)
        self.n, self.start = n, start

    def sample(self, mask: Tensor | None = None) -> Tensor:
        if mask is not None:
            idxes = torch.where(mask)[0].to(dtype=self.dtype)
            if len(idxes) == 0:
                return torch.tensor(self.start, dtype=self.dtype, device=self.device)
            else:
                chosen = torch.randint(
                    high=len(idxes),
                    size=[],
                    device=self.device,
                    generator=self.gen,
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

    def __eq__(self, other):
        if not isinstance(other, TensorDiscrete):
            return False
        return (
            self.n == other.n
            and self.device == other.device
            and self.dtype == other.dtype
        )

    def __repr__(self) -> str:
        if self.start == 0:
            return f"TensorDiscrete({self.n})"
        else:
            return f"TensorDiscrete({self.n}, start={self.start})"
