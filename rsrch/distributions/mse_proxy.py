from torch import Size, Tensor
from torch._C import Size

from rsrch.types import Tensorlike

from .distribution import Distribution
from .utils import sum_rightmost


class MSEProxy(Distribution, Tensorlike):
    """A "proxy" distribution, which is always the same value and for which
    log-prob is MSE. Useful if one wants to mix deterministic and stochastic code."""

    def __init__(self, value: Tensorlike, event_dims: int):
        pivot = len(value.shape) - event_dims
        batch_shape = value.shape[:pivot]
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = value.shape[pivot:]
        self.value = self.register("value", value)

    @property
    def mean(self):
        return self.value

    @property
    def mode(self):
        return self.value

    def log_prob(self, value: Tensor) -> Tensor:
        neg_loss = -0.5 * (value - self.value).square()
        return sum_rightmost(neg_loss, len(self.event_shape))

    def rsample(self, sample_shape: Size = ...) -> Tensor:
        return self.value.expand(*sample_shape, *self.value.shape)
