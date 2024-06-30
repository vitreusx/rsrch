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
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        event_dims=0,
    ):
        if probs is not None and logits is not None:
            raise ValueError("probs and logits cannot be both not-None")

        if probs is not None:
            param_type = "probs"
            param = probs
        else:
            param_type = "logits"
            param = logits

        pivot = len(param.shape) - event_dims
        batch_shape, event_shape = param.shape[:pivot], param.shape[pivot:]

        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        self._param = self.register("_param", param)
        self._param_type = param_type

        self.reset_aux()

    def reset_aux(self):
        if self._param_type == "probs":
            self._probs, self._logits = self._param, None
        else:
            self._probs, self._logits = None, self._param

    def _new(self, shape: torch.Size, fields: dict):
        new = super()._new(shape, fields)
        new.reset_aux()
        return new

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            eps = torch.finfo(self._probs.dtype).eps
            probs = self._probs.clamp(eps, 1 - eps)
            logits = torch.log(probs) - torch.log1p(-probs)
            self._logits = logits
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            self._probs = F.sigmoid(self._logits)
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

    def rsample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        value = value.type_as(self.logits)
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
