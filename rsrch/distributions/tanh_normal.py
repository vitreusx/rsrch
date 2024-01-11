from numbers import Number

from torch import Tensor

from rsrch import spaces

from . import transforms as T
from .normal import Normal
from .transformed import TransformedDistribution


class TanhNormal(TransformedDistribution):
    def __init__(
        self,
        loc: Number | Tensor,
        scale: Number | Tensor,
        box: spaces.torch.Box,
    ):
        super().__init__(
            Normal(loc, scale, len(box.shape)),
            T.TanhTransform(eps=1e-6),
            T.AffineTransform(
                loc=0.5 * (box.low + box.high),
                scale=0.5 * (box.high - box.low),
            ),
        )
