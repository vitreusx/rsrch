import torch
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .kl import register_kl


class Dirac(Distribution, Tensorlike):
    def __init__(self, value: Tensor, event_dims: int):
        batch_shape = value.shape[:-event_dims]
        Tensorlike.__init__(self, batch_shape)

        self.value: Tensor
        self.register("value", value)

    @property
    def mean(self):
        return self.value

    @property
    def mode(self):
        return self.value

    @property
    def variance(self):
        return 0.0

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape=torch.Size()):
        return self.value.expand(*sample_shape, *self.value.shape)

    def log_prob(self, other):
        value = self.value.expand(other.shape)
        return torch.where(value == other, 0.0, -torch.inf)

    def entropy(self):
        return 0.0


@register_kl(Dirac, Distribution)
def _kl_dirac(p: Dirac, q: Distribution):
    return -q.log_prob(p.value)
