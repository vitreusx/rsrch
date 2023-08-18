import math
from numbers import Number

import torch
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .kl import register_kl
from .utils import sum_rightmost


def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


class Normal(Distribution, Tensorlike):
    MSE_SIGMA = 1.0 / math.sqrt(2.0 * math.pi)
    """This value of \sigma for the normal distribution makes log prob equal 
    (to a scale factor) to (x-\mu)^2. """

    def __init__(
        self,
        loc: Number | Tensor,
        scale: Number | Tensor,
        event_dims: int = 0,
    ):
        loc = torch.as_tensor(loc)
        scale = torch.as_tensor(scale, device=loc.device)
        loc, scale = torch.broadcast_tensors(loc, scale)

        bcast_shape = loc.shape
        split_idx = len(bcast_shape) - event_dims
        batch_shape, event_shape = bcast_shape[:split_idx], bcast_shape[split_idx:]

        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        self.loc: Tensor
        self.register("loc", loc)

        self.scale: Tensor
        self.register("scale", scale)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    def sample(self, sample_shape: torch.Size = torch.Size()):
        shape = torch.Size([*sample_shape, *self.batch_shape, *self.event_shape])
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        shape = torch.Size([*sample_shape, *self.batch_shape, *self.event_shape])
        eps = _standard_normal(shape, self.loc.dtype, self.loc.device)
        return self.loc + self.scale * eps

    def log_prob(self, value: Tensor):
        var = self.scale**2
        log_scale = self.scale.log()
        logp = (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ent = 0.5 + 0.5 * math.log(2 * math.pi) + self.scale.log()
        return sum_rightmost(ent, len(self.event_shape))


@register_kl(Normal, Normal)
def _kl_normal(p: Normal, q: Normal):
    var_p, var_q = p.scale.square(), q.scale.square()
    ratio = var_p / var_q
    k = max(len(p.event_shape), 1)
    kl_ = (p.loc - q.loc).square() / (2 * var_q) + (ratio - 1 - ratio.log()) / 2
    return sum_rightmost(kl_, k)
