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
        ind_rv = Categorical(probs=probs, logits=logits)
        Tensorlike.__init__(self, ind_rv.shape)
        self.event_shape = (ind_rv.num_events,)
        self.ind_rv = self.register("ind_rv", ind_rv)

    @property
    def mode(self):
        value = self.ind_rv._param.argmax(-1)
        value = F.one_hot(value, self.ind_rv.num_events)
        value = value.type_as(self.ind_rv._param)
        return value

    def sample(self, sample_shape=()):
        indices = self.ind_rv.sample(sample_shape)
        value = F.one_hot(indices, self.ind_rv.num_events)
        value = value.type_as(self.ind_rv._param)
        return value

    def rsample(self, sample_shape=()):
        value = self.sample(sample_shape)
        probs = self.ind_rv.probs
        probs = probs.expand_as(value)
        value = pass_gradient(value, probs)
        return value

    def log_prob(self, value: Tensor):
        return self.ind_rv.log_prob(value)

    def entropy(self):
        return self.ind_rv.entropy()


@register_kl(OneHot, OneHot)
def _(p: OneHot, q: OneHot):
    return kl_divergence(p.ind_rv, q.ind_rv)


class Discrete(Distribution, Tensorlike):
    """Distribution over tensors of shape [N*D], where each of N chunks
    ("tokens") is a one-hot vector. Used in DreamerV2."""

    def __init__(
        self,
        *,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        onehot = OneHot(probs=probs, logits=logits)
        Tensorlike.__init__(self, onehot.shape[:-1])
        self.num_tokens = onehot.shape[-1]
        self.token_size = onehot.event_shape[0]
        self.onehot = self.register("onehot", onehot)

    @property
    def mode(self):
        value = self.onehot.mode
        return value.flatten(-2)

    def sample(self, sample_shape=()):
        value = self.onehot.sample(sample_shape)
        return value.flatten(-2)

    def rsample(self, sample_shape=()):
        value = self.onehot.rsample(sample_shape)
        return value.flatten(-2)

    def log_prob(self, value: Tensor):
        value = value.reshape(*value.shape[:-1], self.num_tokens, self.token_size)
        value = self.onehot.log_prob(value)
        return value.sum(-1)

    def entropy(self):
        value = self.onehot.entropy()
        return value.sum(-1)


@register_kl(Discrete, Discrete)
def _(p: Discrete, q: Discrete):
    return kl_divergence(p.onehot.ind_rv, q.onehot.ind_rv)
