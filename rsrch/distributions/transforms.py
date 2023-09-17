import math
from numbers import Number
from typing import List, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import sum_rightmost


class Transform(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        ...

    def inv(self, y: Tensor) -> Tensor:
        ...

    def log_abs_det_jac(self, x: Tensor, y: Tensor) -> Tensor:
        ...


class TanhTransform(Transform):
    def __init__(self, eps=None):
        self.eps = eps

    def __call__(self, x):
        y = x.tanh()
        if self.eps is not None:
            y = y.clamp(-1.0 + self.eps, 1.0 - self.eps)
        return y

    def inv(self, y):
        if self.eps is not None:
            y = y.clamp(-1.0 + self.eps, 1.0 - self.eps)
        return y.atanh()

    def log_abs_det_jac(self, x, y):
        return 2 * (math.log(2) - x - F.softplus(-2 * x))


class AffineTransform(Transform):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        if isinstance(loc, Number) and isinstance(scale, Number):
            self._event_dim = 0
        else:
            _param = loc if isinstance(loc, Tensor) else scale
            self._event_dim = len(_param.shape)

    def __call__(self, x):
        return self.loc + x * self.scale

    def inv(self, y):
        return (y - self.loc) / self.scale

    def log_abs_det_jac(self, x, y):
        if not isinstance(self.scale, Tensor):
            res = torch.full_like(x, math.log(abs(self.scale)))
        else:
            res = self.scale.abs().log().expand(x.shape)
        return sum_rightmost(res, self._event_dim)
