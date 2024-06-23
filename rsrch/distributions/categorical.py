import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .kl import register_kl
from .utils import sum_rightmost


class Categorical(Distribution, Tensorlike):
    def __init__(
        self,
        *,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        log_probs: Tensor | None = None,
        event_dims: int = 0,
    ):
        if sum(x is not None for x in (probs, logits, log_probs)) != 1:
            raise ValueError("cannot supply more than 1 param")

        if probs is not None:
            probs = torch.as_tensor(probs)
            _param = probs
            norm = False
        elif logits is not None:
            logits = torch.as_tensor(logits)
            norm = False
            _param = logits
        elif log_probs is not None:
            logits = torch.as_tensor(log_probs)
            norm = True
            _param = logits = log_probs

        batch_shape = _param.shape[: -event_dims - 1]
        event_shape = _param.shape[-event_dims - 1 : -1]
        Tensorlike.__init__(self, batch_shape)

        self.event_shape = event_shape
        self.num_events = _param.shape[-1]

        self._probs = self.register("_probs", probs)
        self._logits = self.register("_logits", logits)
        self._normalized = norm

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            eps = torch.finfo(self._probs.dtype).eps
            probs = self._probs.clamp(eps, 1 - eps)
            self._logits = torch.log(probs)
            self._normalized = True
        return self._logits

    @property
    def _param(self):
        return self._logits if self._logits is not None else self._probs

    @property
    def log_probs(self) -> Tensor:
        if self._logits is None or not self._normalized:
            self._logits = self.logits - self.logits.logsumexp(-1, keepdim=True)
            self._normalized = True
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            if self._normalized:
                self._probs = self._logits.exp()
            else:
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

    def sample(self, sample_shape=()) -> Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self.num_events)
        if sample_shape.numel() == 1:
            samples_2d = self._multinomial1(probs_2d).T
        else:
            samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape([*sample_shape, *self.batch_shape, *self.event_shape])

    def rsample(self, sample_shape=()):
        raise NotImplementedError

    def _multinomial1(self, probs_2d: torch.Tensor):
        q = torch.empty_like(probs_2d).exponential_(1)
        q = probs_2d / q
        return q.argmax(dim=-1, keepdim=True)

    def log_prob(self, value: Tensor) -> Tensor:
        if value.dtype.is_floating_point:
            # Value is a set of one-hot vectors
            return sum_rightmost(value * self.log_probs, len(self.event_shape) + 1)
        else:
            # Value is a set of indices
            value = value.long().unsqueeze(-1)
            value, log_pmf = torch.broadcast_tensors(value, self.log_probs)
            value = value[..., :1]
            logp = log_pmf.gather(-1, value).squeeze(-1)
            return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        log_probs = torch.clamp(self.log_probs, min=min_real)
        p_log_p = log_probs * self.probs
        return -sum_rightmost(p_log_p, len(self.event_shape) + 1)


@register_kl(Categorical, Categorical)
def _(p: Categorical, q: Categorical):
    t = p.probs * (p.log_probs - q.log_probs)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
