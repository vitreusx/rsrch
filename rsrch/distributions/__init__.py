from typing import Optional

import torch
from torch import Tensor
from torch.distributions import *

from rsrch.utils.detach import detach, register_detach

from .categorical import *
from .mutlihead_ohst import *
from .normal import *
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
