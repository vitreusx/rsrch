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
        elif logits is not None:
            param_type = "logits"
            param = logits
        elif log_probs is not None:
            param_type = "log_probs"
            param = log_probs

        batch_shape = param.shape[: -event_dims - 1]
        event_shape = param.shape[-event_dims - 1 : -1]
        Tensorlike.__init__(self, shape=batch_shape)

        self.event_shape = event_shape
        self.num_events = param.shape[-1]

        self._param_type = param_type
        self._param = self.register("_param", param)

    @cached_property
    def logits(self) -> Tensor:
        if self._param_type == "logits":
            logits = self._param
        else:
            logits = self.log_probs
        return logits

    @cached_property
    def log_probs(self) -> Tensor:
        if self._param_type == "log_probs":
            log_probs = self._param
        elif self._param_type == "logits":
            logits = self._param
            log_probs = logits - logits.logsumexp(-1, keepdim=True)
        else:
            probs = self._param
            log_probs = probs.log()
        return log_probs

    @cached_property
    def probs(self) -> Tensor:
        if self._param_type == "probs":
            probs = self._param
        elif self._param_type == "logits":
            logits = self._param
            probs = F.softmax(logits, -1)
        else:
            log_probs = self._param
            probs = log_probs.exp()
        return probs

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
        eps = torch.finfo(logits.dtype).eps
        unif = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        return (logits - (-unif.log()).log()).argmax(-1)

    def rsample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value: Tensor) -> Tensor:
        # value = value.long().unsqueeze(-1)
        # value, log_pmf = torch.broadcast_tensors(value, self.log_probs)
        # value = value[..., :1]
        # logp = log_pmf.gather(-1, value).squeeze(-1)
        # return sum_rightmost(logp, len(self.event_shape))
        logits = self.logits.expand(*value.shape, self.num_events)
        ce = F.cross_entropy(
            logits.reshape(-1, self.num_events),
            value.ravel(),
            reduction="none",
        )
        ce = ce.reshape(value.shape)
        return -sum_rightmost(ce, len(self.event_shape))

    def entropy(self):
        log_p = self.log_probs
        eps = torch.finfo(log_p.dtype).eps
        log_neg_p_log_p = log_p + (-log_p).clamp(min=eps).log()
        log_ent = torch.logsumexp(log_neg_p_log_p, dim=-1)
        return sum_rightmost(log_ent.exp(), len(self.event_shape))


@register_kl(Categorical, Categorical)
def _(p: Categorical, q: Categorical):
    kl_div = F.kl_div(q.log_probs, p.log_probs, reduction="none", log_target=True)
    return sum_rightmost(kl_div, len(p.event_shape) + 1)
