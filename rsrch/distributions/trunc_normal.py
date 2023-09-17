from numbers import Number

import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .distribution import Distribution


def norm_pdf(x, loc=0.0, scale=1.0):
    x = (x - loc) / scale
    return x**2


class TruncNormal(Tensorlike, Distribution):
    """Normal distribution truncated to [-1, 1]^d."""

    def __init__(self, loc: Tensor | Number, scale: Tensor | Number, event_dims=0):
        loc = torch.as_tensor(loc)
        scale = torch.as_tensor(scale, device=loc.device)
        loc, scale = torch.broadcast_tensors(loc, scale)

        bcast_shape = loc.shape
        split_idx = len(bcast_shape) - event_dims
        batch_shape, event_shape = bcast_shape[:split_idx], bcast_shape[split_idx:]

        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        self.loc: Tensor
        self.register("loc", loc)

        self.scale: Tensor
        self.register("scale", scale)
