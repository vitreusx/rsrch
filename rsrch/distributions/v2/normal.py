import math
from numbers import Number

import torch
from tensordict import tensorclass
from torch import Tensor

from .kl import register_kl
from .utils import _sum_rightmost, distribution


def _standard_normal(shape, dtype, device):
    if torch._C._get_tracing_state():
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


@distribution
class Normal:
    loc: Tensor
    scale: Tensor
    event_shape: torch.Size

    def __init__(
        self,
        loc: Number | Tensor,
        scale: Number | Tensor,
        event_dims: int = 0,
    ):
        loc = torch.as_tensor(loc)
        scale = torch.as_tensor(scale)
        loc, scale = torch.broadcast_tensors(loc, scale)

        split_idx = len(loc.shape) - event_dims
        batch_shape, event_shape = loc.shape[:split_idx], loc.shape[split_idx:]

        self.__tc_init__(loc, scale, event_shape, batch_size=batch_shape)

    @property
    def batch_shape(self):
        return self.batch_size

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
        return _sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ent = 0.5 + 0.5 * math.log(2 * math.pi) + self.scale.log()
        return _sum_rightmost(ent, len(self.event_shape))

    # def __repr__(self):
    #     return f"Normal(batch_shape: {[*self.batch_shape]}, event_shape: {[*self.event_shape]})"


@register_kl(Normal, Normal)
def _kl_normal(p: Normal, q: Normal):
    var_p, var_q = p.scale.square(), q.scale.square()
    ratio = var_p / var_q
    k = len(p.event_shape)
    kl_ = (p.loc - q.loc).square() / (2 * var_q) + (ratio - k - ratio.log()) / 2
    return _sum_rightmost(kl_, k)
