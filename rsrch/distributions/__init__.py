from typing import Optional

import torch
from torch import Tensor
from torch.distributions import *

from rsrch.utils.detach import detach, register_detach

from .categorical import *
from .one_hot_categorical import *
from .transforms import *


@register_detach(Categorical)
def _detach_cat(rv: Categorical):
    if "logits" in rv.__dict__:
        logits = rv.logits.detach()
        return Categorical(logits=logits, validate_args=rv._validate_args)
    else:
        probs = rv.probs.detach()
        return Categorical(probs=probs, validate_args=rv._validate_args)


@register_detach(Normal)
def _detach_normal(rv: Normal):
    return Normal(
        loc=rv.loc.detach(),
        scale=rv.scale.detach(),
        validate_args=rv._validate_args,
    )


@register_detach(Independent)
def _detach_ind(rv: Independent):
    return Independent(
        base_distribution=detach(rv.base_dist),
        reinterpreted_batch_ndims=rv.reinterpreted_batch_ndims,
        validate_args=rv._validate_args,
    )


@register_detach(TransformedDistribution)
def _detach_transformed(rv: TransformedDistribution):
    return TransformedDistribution(
        base_distribution=detach(rv.base_dist),
        transforms=rv.transforms,
        validate_args=rv._validate_args,
    )


class MultiheadOHST(Distribution):
    arg_constraints = {}

    def __init__(
        self,
        enc_dim: int,
        num_classes: int,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        self.enc_dim = enc_dim
        assert enc_dim % num_classes == 0
        num_heads = enc_dim // num_classes
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.probs = probs
        self.logits = logits

        batch_shape, event_shape = torch.Size([]), torch.Size([])
        if probs is not None:
            batch_shape, event_shape = probs.shape[:-1], probs.shape[-1:]
        elif logits is not None:
            batch_shape, event_shape = logits.shape[:-1], logits.shape[-1:]

        assert event_shape[0] == num_heads * num_classes
        self._batch_shape = batch_shape
        self._event_shape = event_shape

        OHST = OneHotCategoricalStraightThrough
        if logits is not None:
            self._unwrapped = OHST(logits=self._split(logits))
        else:
            self._unwrapped = OHST(probs=self._split(probs))
        self._base = Independent(self._unwrapped, 1)

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


@register_kl(MultiheadOHST, MultiheadOHST)
def _multihead_ohst_kl(p: MultiheadOHST, q: MultiheadOHST):
    return kl_divergence(p._base, q._base)


@register_detach(MultiheadOHST)
def _detach_mh_ohst(rv: MultiheadOHST):
    if rv.logits is not None:
        kw = dict(logits=rv.logits.detach())
    else:
        kw = dict(probs=rv.probs.detach())
    return MultiheadOHST(rv.enc_dim, rv.num_classes, **kw)
