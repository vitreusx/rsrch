from numbers import Number

from torch import Tensor

from . import transforms as T
from .normal import Normal
from .transformed import TransformedDistribution


class SquashedNormal(TransformedDistribution):
    def __init__(
        self,
        loc: Number | Tensor,
        scale: Number | Tensor,
        low: Number | Tensor,
        high: Number | Tensor,
        event_dims: int = 0,
    ):
        super().__init__(
            Normal(loc, scale, event_dims),
            T.TanhTransform(eps=1e-6),
            T.AffineTransform(
                loc=0.5 * (low + high),
                scale=0.5 * (high - low),
            ),
        )
