from .transformed import TransformedDistribution
from . import transforms as T
import torch
from torch import Tensor
import torch.nn.functional as F
from .normal import Normal


class TanhNormal(TransformedDistribution):
    def __init__(self, param: Tensor, event_dims=1, scale=5.0, min_std=0.0):
        mean, std = param.chunk(2, 1)
        mean = scale * torch.tanh(mean / scale)
        std = F.softplus(std) + min_std
        super().__init__(
            Normal(mean, std, event_dims),
            T.TanhTransform(),
        )
