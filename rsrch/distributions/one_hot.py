import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch.nn.utils import pass_gradient
from rsrch.types.tensorlike import Tensorlike

from .categorical import Categorical
from .distribution import Distribution
from .kl import kl_divergence, register_kl


class OneHot(Distribution, Tensorlike):
    def __init__(
        self,
        *,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        index_dist = Categorical(probs=probs, logits=logits)
        Tensorlike.__init__(self, index_dist.shape)
        self.event_shape = (index_dist.num_events,)
        self.index_dist = self.register("index_dist", index_dist)

    @property
    def mode(self):
        value = self.index_dist._param.argmax(-1)
        value = F.one_hot(value, self.index_dist.num_events)
        value = value.type_as(self.index_dist._param)
        return value

    def sample(self, sample_shape=()):
        indices = self.index_dist.sample(sample_shape)
        value = F.one_hot(indices, self.index_dist.num_events)
        value = value.type_as(self.index_dist._param)
        return value

    def rsample(self, sample_shape=()):
        value = self.sample(sample_shape)
        probs = self.index_dist.probs
        probs = probs.expand_as(value)
        value = pass_gradient(value, probs)
        return value

    def log_prob(self, value: Tensor):
        # indices = value.argmax(-1)
        # return self.index_dist.log_prob(indices)
        logits = self.index_dist.logits
        logits = logits.expand(value.shape)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            value.reshape(-1, value.shape[-1]),
            reduction="none",
        )
        ce = ce.reshape(value.shape[:-1])
        return -ce

    def entropy(self):
        return self.index_dist.entropy()


@register_kl(OneHot, OneHot)
def _(p: OneHot, q: OneHot):
    return kl_divergence(p.index_dist, q.index_dist)
