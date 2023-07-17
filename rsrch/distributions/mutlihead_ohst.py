from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, kl_divergence, register_kl
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

from rsrch.utils.detach import register_detach

from .one_hot_categorical import OneHotCategoricalStraightThrough


class MultiheadOHST(Distribution):
    arg_constraints = {}

    def __init__(
        self,
        enc_dim: int,
        num_classes: int,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args=False,
    ):
        self.enc_dim = enc_dim
        assert enc_dim % num_classes == 0
        num_heads = enc_dim // num_classes
        self.num_heads = num_heads
        self.num_classes = num_classes

        if probs is not None:
            self.probs = probs
        else:
            self.logits = logits

        batch_shape, event_shape = torch.Size([]), torch.Size([])
        if probs is not None:
            batch_shape, event_shape = probs.shape[:-1], probs.shape[-1:]
        elif logits is not None:
            batch_shape, event_shape = logits.shape[:-1], logits.shape[-1:]

        assert event_shape[0] == num_heads * num_classes
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        OHST = OneHotCategoricalStraightThrough
        if logits is not None:
            self._unwrapped = OHST(logits=self._split(logits))
        else:
            self._unwrapped = OHST(probs=self._split(probs))
        self._base = Independent(self._unwrapped, 1)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    def log_prob(self, value: Tensor) -> Tensor:
        return self._base.log_prob(self._split(value))

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._merge(self._base.sample(sample_shape))

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        return self._merge(self._base.rsample(sample_shape))

    def _merge(self, x: Tensor):
        return x.reshape(*x.shape[:-2], self.enc_dim)

    def _split(self, x: Tensor):
        return x.reshape(*x.shape[:-1], self.num_heads, self.num_classes)

    @classmethod
    def _torch_cat(
        cls, dists: List[MultiheadOHST] | Tuple[MultiheadOHST, ...], dim: int = 0
    ):
        dist = dists[0]
        args = dict(
            enc_dim=dist.enc_dim,
            num_classes=dist.num_classes,
            validate_args=dist._validate_args,
        )
        dim = range(len(dist.batch_shape)).index(dim)
        if "probs" in dist.__dict__:
            probs = torch.cat([d.probs for d in dists], dim)
            args.update(probs=probs)
        else:
            logits = torch.cat([d.logits for d in dists], dim)
            args.update(logits=logits)
        return cls(**args)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(t, MultiheadOHST) for t in types):
            return NotImplemented

        if func == torch.cat:
            return cls._torch_cat(*args, **kwargs)

        return NotImplemented


@register_kl(MultiheadOHST, MultiheadOHST)
def _multihead_ohst_kl(p: MultiheadOHST, q: MultiheadOHST):
    return kl_divergence(p._base, q._base)


@register_detach(MultiheadOHST)
def _detach_mh_ohst(rv: MultiheadOHST):
    args = dict(
        enc_dim=rv.enc_dim,
        num_classes=rv.num_classes,
        validate_args=rv._validate_args,
    )
    if rv.logits is not None:
        args.update(logits=rv.logits.detach())
    else:
        args.update(probs=rv.probs.detach())
    return MultiheadOHST(**args)
