import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dist_head as dh


def layer_init(layer, std=nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
    return layer


class Beta(nn.Module):
    def __init__(self, in_features: int, out_space: spaces.torch.Box):
        super().__init__()
        self.shape = out_space.shape
        out_dim = int(np.prod(out_space.shape))
        self.fc = nn.Linear(in_features, 2 * out_dim)
        layer_init(self.fc, std=1e-2)
        self.register_buffer("loc", out_space.low)
        self.register_buffer("scale", out_space.high - out_space.low)

    def forward(self, x: Tensor):
        out: Tensor = self.fc(x)
        alpha, beta = out.chunk(2, -1)
        alpha, beta = 1.0 + alpha.exp(), 1.0 + beta.exp()
        alpha, beta = alpha.reshape(-1, *self.shape), beta.reshape(-1, *self.shape)
        return D.Affine(
            D.Beta(alpha, beta, len(self.shape)),
            self.loc,
            self.scale,
        )


class ClipNormal(nn.Module):
    def __init__(self, in_features: int, out_space: spaces.torch.Box):
        super().__init__()
        self.shape = out_space.shape
        out_dim = int(np.prod(out_space.shape))
        self.fc = nn.Linear(in_features, 2 * out_dim)
        layer_init(self.fc, std=1e-2)
        self.register_buffer("loc", 0.5 * (out_space.low + out_space.high))
        self.register_buffer("scale", 0.5 * (out_space.high - out_space.low))

    def forward(self, x: Tensor):
        out: Tensor = self.fc(x)
        mean, logstd = out.chunk(2, -1)
        mean = self.loc + self.scale * mean
        std = F.softplus(logstd) * self.scale
        return D.ClipNormal(
            D.Normal(mean, std, len(self.shape)),
            self.out_space,
        )


def SpaceDistLayer(in_features: int, out_space: spaces.torch.Space):
    """A final network layer parametrizing distribution over values in a given
    space."""

    if isinstance(out_space, spaces.torch.Box):
        if out_space.bounded.all():
            return Beta(in_features, out_space)
        else:
            return ClipNormal(in_features, out_space)

    elif isinstance(out_space, spaces.torch.Discrete):
        return dh.Categorical(in_features, out_space.n)
