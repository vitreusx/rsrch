import torch
import torch.nn.functional as F
from torch import Tensor

from .distribution import Distribution
from .kl import register_kl
from .tensorlike import Tensorlike


class Categorical(Distribution, Tensorlike):
    def __init__(self, probs: Tensor | None = None, logits: Tensor | None = None):
        if probs is not None and logits is not None:
            raise ValueError("probs and logits cannot be both not-None")

        _param = probs if probs is not None else logits
        shape = _param.shape[:-1]
        Tensorlike.__init__(self, shape)

        self._probs: Tensor | None
        self.register_field("_probs", probs)

        self._logits: Tensor | None
        self.register_field("_logits", logits)

        self._normalized = False
        self._num_events = _param.shape[-1]
        self.event_shape = torch.Size([])

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            eps = torch.finfo(self._probs.dtype).eps
            probs = self._probs.clamp(eps, 1 - eps)
            self._logits = torch.log(probs)
            self.__tensor_fields__.append("_logits")
            self._normalized = True
        return self._logits

    @property
    def log_probs(self) -> Tensor:
        if self._logits is None or not self._normalized:
            self._logits = self.logits - self.logits.logsumexp(-1, keepdim=True)
            self._normalized = True
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            self._probs = F.softmax(self.logits, -1)
            self.__tensor_fields__.append("_probs")
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


@register_kl(Categorical, Categorical)
def _kl_categorical(p: Categorical, q: Categorical):
    t = p.probs * (p.log_probs - q.log_probs)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
