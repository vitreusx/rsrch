import torch
import torch.distributions as D
from torch import Tensor
from torch.distributions import *

from rsrch.utils.detach import detach, register_detach

from .transforms import *


@register_detach(D.Categorical)
def _detach_cat(rv: D.Categorical):
    if "logits" in rv.__dict__:
        logits = rv.logits.detach()
        return D.Categorical(logits=logits, validate_args=rv._validate_args)
    else:
        probs = rv.probs.detach()
        return D.Categorical(probs=probs, validate_args=rv._validate_args)


@register_detach(D.Normal)
def _detach_normal(rv: D.Normal):
    return D.Normal(
        loc=rv.loc.detach(),
        scale=rv.scale.detach(),
        validate_args=rv._validate_args,
    )


@register_detach(D.Independent)
def _detach_ind(rv: D.Independent):
    return D.Independent(
        base_distribution=detach(rv.base_dist),
        reinterpreted_batch_ndims=rv.reinterpreted_batch_ndims,
        validate_args=rv._validate_args,
    )


@register_detach(D.TransformedDistribution)
def _detach_transformed(rv: D.TransformedDistribution):
    return D.TransformedDistribution(
        base_distribution=detach(rv.base_dist),
        transforms=rv.transforms,
        validate_args=rv._validate_args,
    )


class MultiheadOHST(D.Distribution):
    arg_constraints = {}

    def __init__(
        self,
        enc_dim: int,
        num_classes: int,
        *,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
    ):
        self.enc_dim = enc_dim
        assert enc_dim % num_classes == 0
        num_heads = enc_dim // num_classes
        self.num_heads = num_heads
        self.num_classes = num_classes
        if logits is not None:
            self.logits = logits
        else:
            self.probs = probs

        x: Tensor = probs if probs is not None else logits
        batch_shape, event_shape = x.shape[:-1], x.shape[-1]
        assert event_shape == num_heads * num_classes
        super().__init__(batch_shape, event_shape)

        OHST = D.OneHotCategoricalStraightThrough
        if logits is not None:
            self._unwrapped = OHST(logits=self._split(logits))
        else:
            self._unwrapped = OHST(probs=self._split(probs))
        self._base = D.Independent(self._unwrapped, 1)

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


@D.register_kl(MultiheadOHST, MultiheadOHST)
def _multihead_ohst_kl(p: MultiheadOHST, q: MultiheadOHST):
    return D.kl_divergence(p._base, q._base)


@register_detach(MultiheadOHST)
def _detach_mh_ohst(rv: MultiheadOHST):
    if "logits" in rv.__dict__:
        kw = dict(logits=rv.logits.detach())
    else:
        kw = dict(probs=rv.probs.detach())
    return MultiheadOHST(rv.enc_dim, rv.num_classes, **kw)
