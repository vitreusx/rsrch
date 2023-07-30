import torch
import torch.nn.functional as F
from torch import Tensor

from .categorical import Categorical
from .kl import register_kl
from .utils import distribution


@distribution
class OneHotCategorical:
    index_rv: Categorical
    event_shape: torch.Size

    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None):
        index_rv = Categorical(probs=probs, logits=logits)
        event_shape = torch.Size([index_rv._num_events])
        self.__tc_init__(
            index_rv=index_rv,
            event_shape=event_shape,
            batch_size=index_rv.batch_size,
        )

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def logits(self):
        return self.index_rv.logits

    @property
    def log_probs(self):
        return self.index_rv.log_probs

    @property
    def probs(self):
        return self.index_rv.probs

    @property
    def mean(self):
        return self.index_rv.probs

    @property
    def mode(self):
        mode = self.probs.argmax(axis=-1)
        mode = F.one_hot(mode, num_classes=self.index_rv._num_events)
        mode = mode.type_as(self.probs)
        return mode

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        num_events = self.event_shape[0]
        indices = self.index_rv.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).type_as(self.probs)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return NotImplemented

    def log_prob(self, value: Tensor):
        _, indices = value.max(-1)
        return self.index_rv.log_prob(indices)

    def entropy(self):
        return self.index_rv.entropy()

    # def __repr__(self):
    #     return f"OneHotCategorical(batch_shape: {[*self.batch_shape]}, num_events: {self.index_rv._num_events})"


@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical(p: OneHotCategorical, q: OneHotCategorical):
    t = p.probs * (p.log_probs - q.log_probs)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, grad_target):
        return value

    @staticmethod
    def backward(ctx, out_grad):
        return None, out_grad


@distribution
class OneHotCategoricalST:
    index_rv: Categorical
    event_shape: torch.Size

    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None):
        index_rv = Categorical(probs=probs, logits=logits)
        event_shape = torch.Size([index_rv._num_events])
        self.__tc_init__(
            index_rv=index_rv,
            event_shape=event_shape,
            batch_size=index_rv.batch_size,
        )

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def logits(self):
        return self.index_rv.logits

    @property
    def log_probs(self):
        return self.index_rv.log_probs

    @property
    def probs(self):
        return self.index_rv.probs

    @property
    def mean(self):
        return self.index_rv.probs

    @property
    def mode(self):
        mode = self.probs.argmax(axis=-1)
        mode = F.one_hot(mode, num_classes=self.index_rv._num_events)
        mode = mode.type_as(self.probs)
        return mode

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        num_events = self.event_shape[0]
        indices = self.index_rv.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).type_as(self.probs)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        samples = self.sample(sample_shape)
        _param = (
            self.index_rv._logits
            if self.index_rv._logits is not None
            else self.index_rv._probs
        )
        return StraightThrough.apply(samples, _param)

    def log_prob(self, value: Tensor):
        _, indices = value.max(-1)
        return self.index_rv.log_prob(indices)

    def entropy(self):
        return self.index_rv.entropy()

    # def __repr__(self):
    #     return f"OneHotCategoricalST(batch_shape: {[*self.batch_shape]}, num_events: {self.index_rv._num_events})"


@register_kl(OneHotCategoricalST, OneHotCategoricalST)
def _kl_onehotcategoricalst(p: OneHotCategoricalST, q: OneHotCategoricalST):
    t = p.probs * (p.log_probs - q.log_probs)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
