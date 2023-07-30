import torch
from tensordict import TensorDict, tensorclass
from torch import Tensor

from .kl import kl_divergence, register_kl
from .one_hot_categorical import OneHotCategoricalST
from .utils import _sum_rightmost, distribution


@distribution
class MultiheadOHST:
    base: OneHotCategoricalST
    num_heads: int
    head_dim: int
    event_shape: torch.Size

    def __init__(
        self,
        num_heads: int,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        _param = probs if probs is not None else logits
        head_dim = _param.shape[-1] // num_heads
        if probs is not None:
            probs = probs.reshape(*probs.shape[:-1], num_heads, head_dim)
            base = OneHotCategoricalST(probs=probs)
        else:
            logits = logits.reshape(*logits.shape[:-1], num_heads, head_dim)
            base = OneHotCategoricalST(logits=logits)

        batch_shape, event_shape = _param.shape[:-1], _param.shape[-1:]
        self.__tc_init__(
            base,
            num_heads,
            head_dim,
            event_shape,
            batch_size=batch_shape,
        )

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def embed_dim(self):
        return self.event_shape[0]

    def _1d(self, value: Tensor):
        return value.flatten(-2)

    def _2d(self, value: Tensor):
        return value.reshape(*value.shape[:-1], self.num_heads, self.head_dim)

    @property
    def probs(self):
        return self._1d(self.base.probs)

    @property
    def log_probs(self):
        return self._1d(self.base.log_probs)

    @property
    def logits(self):
        return self._1d(self.base.logits)

    def log_prob(self, value: Tensor):
        logp = self.base.log_prob(self._2d(value))
        return _sum_rightmost(logp, 1)

    def entropy(self):
        return _sum_rightmost(self.base.entropy(), 1)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self._1d(self.base.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        return self._1d(self.base.rsample(sample_shape))

    # def __repr__(self):
    #     return f"MultiheadOHST(num_heads: {self.num_heads}, batch_shape: {[*self.batch_shape]}, event_shape: {[*self.event_shape]})"


@register_kl(MultiheadOHST, MultiheadOHST)
def _kl_multihead_ohst(p: MultiheadOHST, q: MultiheadOHST):
    _kl = kl_divergence(p.base, q.base)
    return _sum_rightmost(_kl, 1)
