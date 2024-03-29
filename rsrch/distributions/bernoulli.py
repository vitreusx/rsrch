from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .kl import register_kl


class Bernoulli(Distribution, Tensorlike):
    _probs: Optional[Tensor]
    _logits: Optional[Tensor]
    event_shape: torch.Size

    def __init__(
        self, probs: Tensor | None = None, logits: Tensor | None = None, event_dims=0
    ):
        if probs is not None and logits is not None:
            raise ValueError("probs and logits cannot be both not-None")

        if probs is not None:
            probs = torch.as_tensor(probs)
            _param = probs
        else:
            logits = torch.as_tensor(logits)
            _param = logits

        pivot = len(_param.shape) - event_dims
        batch_shape = _param.shape[:-event_dims]
        event_shape = _param.shape[-event_dims:]

        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        self._probs: Tensor | None
        self.register("_probs", probs)

        self._logits: Tensor | None
        self.register("_logits", logits)

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            eps = torch.finfo(self._probs.dtype).eps
            probs = self._probs.clamp(eps, 1 - eps)
            logits = torch.log(probs) - torch.log1p(-probs)
            self.register("_logits", logits)
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            probs = F.sigmoid(self._logits)
            self.register("_probs", probs)
        return self._probs

    @property
    def mean(self):
        return self.probs

    @property
    def mode(self):
        mode = (self.probs >= 0.5).to(self.probs)
        # mode[self.probs == 0.5] = torch.nan
        return mode

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size([*sample_shape, *self.batch_shape, *self.event_shape])
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        if value.dtype != self.logits.dtype:
            value = value.to(dtype=self.logits.dtype)
        logits, value = torch.broadcast_tensors(self.logits, value)
        return -F.binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self):
        return F.binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )


@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli(p: Bernoulli, q: Bernoulli):
    t1 = p.probs * (F.softplus(-q.logits) - F.softplus(-p.logits))
    t1[q.probs == 0] = torch.inf
    t1[p.probs == 0] = 0
    t2 = (1 - p.probs) * (F.softplus(q.logits) - F.softplus(p.logits))
    t2[q.probs == 1] = torch.inf
    t2[p.probs == 1] = 0
    return t1 + t2
