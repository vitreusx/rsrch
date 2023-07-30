import math

import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Tensor, nn
from torch.distributions.transforms import *


class ConvertDtype(D.Transform):
    domain = C.dependent(event_dim=0)
    codomain = C.dependent(event_dim=0)

    def __init__(self, from_dt: torch.dtype, to_dt: torch.dtype):
        super().__init__()
        self.from_dt = from_dt
        self.to_dt = to_dt

    def __eq__(self, other):
        return (
            isinstance(other, ConvertDtype)
            and self.from_dt == other.from_dt
            and self.to_dt == other.to_dt
        )

    def _call(self, x: Tensor):
        return x.to(dtype=self.to_dt)

    def _inverse(self, y: Tensor):
        return y.to(dtype=self.from_dt)

    def log_abs_det_jacobian(self, x, y):
        return torch.ones([], dtype=self.to_dt).item()


class SafeTanhTransform(D.Transform):
    domain = C.real
    codomain = C.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    eps = 1e-8
    min_v, max_v = -1.0 + eps, 1.0 - eps
    log2 = math.log(2)

    def __eq__(self, other):
        return isinstance(other, SafeTanhTransform)

    def _call(self, x: Tensor):
        return x.tanh()

    def _inverse(self, y: Tensor):
        y = y.clamp(self.min_v, self.max_v)
        return y.atanh()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor):
        return 2.0 * (self.log2 - x - nn.functional.softplus(-2.0 * x))


class SliceTransform(D.Transform):
    domain = C.dependent(event_dim=0)
    codomain = C.dependent(event_dim=0)
    bijective = False

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def __eq__(self, other):
        return isinstance(other, SliceTransform) and self.idx == other.idx

    def _call(self, x):
        return x[self.idx]

    def log_abs_det_jacobian(self, x, y):
        return torch.ones([]).type_as(y)


class Pipeline(D.TransformedDistribution):
    def __init__(self, *parts: D.Distribution | D.Transform, validate_args=False):
        super().__init__(
            base_distribution=parts[0],
            transforms=D.ComposeTransform(parts[1:]),
            validate_args=validate_args,
        )
