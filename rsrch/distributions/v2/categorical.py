from typing import Any

import torch
import torch.nn.functional as F
from tensordict import tensorclass
from torch import Tensor

from .kl import register_kl
from .utils import distribution


@distribution
class Categorical:
    _logits: Any
    _log_probs: Any
    _probs: Any
    event_shape: torch.Size
    _num_events: int

    def __init__(self, probs: Tensor | None = None, logits=None):
        if probs is not None and logits is not None:
            raise ValueError("probs and logits cannot be both not-None")

        _param = probs if probs is not None else logits
        batch_size = _param.shape[:-1]
        event_shape = torch.Size([])
        _num_events = _param.shape[-1]

        self.__tc_init__(
            _logits=logits,
            _log_probs=None,
            _probs=probs,
            event_shape=event_shape,
            _num_events=_num_events,
            batch_size=batch_size,
        )

    @property
    def batch_shape(self):
        return self.batch_size

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            eps = torch.finfo(self._probs.dtype).eps
            probs = self._probs.clamp(eps, 1 - eps)
            self._logits = torch.log(probs)
        return self._logits

    @property
    def log_probs(self) -> Tensor:
        if self._log_probs is None:
            self._log_probs = self.logits - self.logits.logsumexp(-1, keepdim=True)
        return self._log_probs

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            self._probs = F.softmax(self.logits, -1)
        return self._probs

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    @property
    def variance(self):
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        if sample_shape.numel() == 1:
            samples_2d = self._multinomial1(probs_2d).T
        else:
            samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape([*sample_shape, *self.batch_shape, *self.event_shape])

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        raise NotImplementedError

    def _multinomial1(self, probs_2d: torch.Tensor):
        q = torch.empty_like(probs_2d).exponential_(1)
        q = probs_2d / q
        return q.argmax(dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    # def __repr__(self):
    #     return f"Categorical(batch_shape: {[*self.batch_shape]}, num_events: {self._num_events})"


@register_kl(Categorical, Categorical)
def _kl_categorical(p: Categorical, q: Categorical):
    t = p.probs * (p.log_probs - q.log_probs)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
