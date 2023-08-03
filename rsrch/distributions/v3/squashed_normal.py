from numbers import Number

from torch import Tensor

from . import transforms as T
from .normal import Normal
from .transformed import TransformedDistribution


class SquashedNormal(TransformedDistribution):
    def __init__(
        self,
        loc: Tensor | Number,
        scale: Tensor | Number,
        event_dims: int,
        min_v: Tensor | Number,
        max_v: Tensor | Number,
    ):
        super().__init__(
            Normal(loc, scale, event_dims),
            T.TanhTransform(),
            T.AffineTransform((min_v + max_v) / 2, (max_v - min_v) / 2),
        )
