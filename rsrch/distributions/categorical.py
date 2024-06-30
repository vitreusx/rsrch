from functools import cached_property

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
            param_type = "probs"
            param = probs
            normalized = False
        elif logits is not None:
            param_type = "logits"
            param = logits
            normalized = False
        elif log_probs is not None:
            param_type = "log_probs"
            param = log_probs
            normalized = True

        batch_shape = param.shape[: -event_dims - 1]
        event_shape = param.shape[-event_dims - 1 : -1]
        Tensorlike.__init__(self, shape=batch_shape)

        self.event_shape = event_shape
        self.num_events = param.shape[-1]

        self._param_type = param_type
        self._normalized = normalized
        self._param = self.register("_param", param)

        self._reset_aux()

    def _reset_aux(self):
        if self._param_type == "probs":
            self._logits, self._probs = None, self._param
            self._normalized = False
        else:
            self._logits, self._probs = self._param, None
            self._normalized = self._param_type == "log_probs"

    def _new(self, shape: torch.Size, fields: dict):
        new = super()._new(shape, fields)
        # super()._new copies the non-registered fields, including _probs and
        # _logits, which might become "stale". We need to reset them.
        new._reset_aux()
        return new

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            probs = self._param
            eps = torch.finfo(probs.dtype).eps
            probs = probs.clamp(eps, 1 - eps)
            self._logits, self._normalized = probs.log(), True
        return self._logits

    @property
    def log_probs(self) -> Tensor:
        if not self._normalized:
            self._logits = self.logits - self.logits.logsumexp(-1, keepdim=True)
            self.normalized = True
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            self._probs = self.logits.softmax(-1)
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
        # if not isinstance(sample_shape, torch.Size):
        #     sample_shape = (sample_shape,)
        # probs_2d = self.probs.reshape(-1, self.num_events)
        # if sample_shape.numel() == 1:
        #     q = torch.empty_like(probs_2d).exponential_(1)
        #     q = probs_2d / q
        #     samples_2d = q.argmax(dim=-1, keepdim=True)
        # else:
        #     samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        # return samples_2d.reshape([*sample_shape, *self.batch_shape, *self.event_shape])
        logits = self.logits.expand(*sample_shape, *self.logits.shape)
        unif = torch.rand_like(logits).clamp(1e-8, 1.0 - 1e-8)
        return (logits - (-unif.log()).log()).argmax(-1)

    def rsample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
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
