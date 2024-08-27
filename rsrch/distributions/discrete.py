from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .kl import kl_divergence, register_kl
from .one_hot import OneHot


class Discrete(Distribution, Tensorlike):
    """Distribution over tensors of shape [N*D], where each of N chunks
    ("tokens") is a one-hot vector. Used in DreamerV2."""

    def __init__(
        self,
        *,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        base = OneHot(probs=probs, logits=logits)
        Tensorlike.__init__(self, base.shape[:-1])
        self.num_tokens = base.shape[-1]
        self.token_size = base.event_shape[0]
        self.base = self.register("base", base)

    @property
    def mode(self):
        value = self.base.mode
        return value.flatten(-2)

    def sample(self, sample_shape=()):
        value = self.base.sample(sample_shape)
        return value.flatten(-2)

    def rsample(self, sample_shape=()):
        value = self.base.rsample(sample_shape)
        return value.flatten(-2)

    def log_prob(self, value: Tensor):
        value = value.reshape(*value.shape[:-1], self.num_tokens, self.token_size)
        value = self.base.log_prob(value)
        return value.sum(-1)

    def entropy(self):
        value = self.base.entropy()
        return value.sum(-1)


@register_kl(Discrete, Discrete)
def _(p: Discrete, q: Discrete):
    return kl_divergence(p.base, q.base).sum(-1)
