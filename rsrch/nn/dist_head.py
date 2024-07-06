import math
from functools import partial
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces


def layer_init(layer, std=1e-2, bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)


class Normal(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: int | tuple[int, ...],
        std_act: Literal["mse", "softplus", "sigmoid", "sigmoid2"] = "softplus",
        min_std: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        if not isinstance(out_shape, Sequence):
            out_shape = (out_shape,)
        self.out_shape = out_shape
        self.std_act = std_act
        self.min_std = min_std

        mult = 1 if std_act == "mse" else 2
        out_size = int(np.prod(self.out_shape))
        self.fc = nn.Linear(in_features, mult * out_size)

    def forward(self, input: Tensor):
        params: Tensor = self.fc(input)
        if self.std_act == "mse":
            mean = params
            mean = mean.reshape(-1, *self.out_shape)
            std = D.Normal.MSE_SIGMA
        else:
            mean, std = params.chunk(2, -1)
            mean = mean.reshape(-1, *self.out_shape)
            std = std.reshape(-1, *self.out_shape)
            if self.std_act == "softplus":
                beta = 1.0 / self.min_std if self.min_std > 0.0 else 1.0
                std = F.softplus(std, beta)
            elif self.std_act == "sigmoid":
                std = F.sigmoid(std)
            elif self.std_act == "sigmoid2":
                std = 2.0 * F.sigmoid(0.5 * std)
        if self.min_std > 0.0:
            std = std + self.min_std
        return D.Normal(mean, std, len(self.out_shape))


class Bernoulli(nn.Module):
    def __init__(
        self,
        in_features: int,
        uniform_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features

        self.fc = nn.Linear(in_features, 1)
        if uniform_init:
            self.fc.apply(partial(layer_init, bias=0.0, std=1e-2))

    def forward(self, x: Tensor) -> D.Bernoulli:
        x = self.fc(x).ravel()
        return D.Bernoulli(logits=x)


class Categorical(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        uniform_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Linear(in_features, num_classes)
        if uniform_init:
            self.fc.apply(partial(layer_init, bias=0.0, std=1e-2))

    def forward(self, x: Tensor) -> D.Categorical:
        x = self.fc(x)
        return D.Categorical(logits=x)


class Dirac(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        self.in_features = in_features
        if isinstance(out_shape, int):
            out_shape = (out_shape,)
        self.out_shape = out_shape

        self.fc = nn.Linear(in_features, math.prod(out_shape))

    def forward(self, x: Tensor) -> D.Dirac:
        out = self.fc(x).reshape(-1, *self.out_shape)
        return D.Dirac(out, len(self.out_shape))


class Beta(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_space: spaces.torch.Box,
        like_normal: bool = True,
        uniform_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.like_normal = like_normal

        self._shape = out_space.shape
        self._bias = 1.0 if like_normal else 0.0

        self.fc = nn.Linear(in_features, 2 * int(np.prod(out_space.shape)))
        if uniform_init:
            # Make bias + alpha.exp() and bias + beta.exp() close to one
            if like_normal:
                self.fc.apply(partial(layer_init, std=1.0, bias=-5.0))
            else:
                self.fc.apply(partial(layer_init, std=1e-2, bias=0.0))

        self.loc: Tensor
        self.register_buffer("loc", out_space.low)
        self.scale: Tensor
        self.register_buffer("scale", out_space.high - out_space.low)

    def forward(self, x: Tensor):
        out: Tensor = self.fc(x)
        alpha, beta = out.chunk(2, -1)
        alpha, beta = self._bias + alpha.exp(), self._bias + beta.exp()
        alpha, beta = alpha.reshape(-1, *self._shape), beta.reshape(-1, *self._shape)
        dist = D.Beta(alpha, beta, len(self._shape))
        dist = D.Affine(dist, self.loc, self.scale)
        return dist


class Discrete(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        num_tokens: int,
        token_size: int,
        uniform_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_tokens = num_tokens
        self.token_size = token_size

        self.fc = nn.Linear(in_features, self.num_tokens * self.token_size)
        if uniform_init:
            self.fc.apply(partial(layer_init, std=1e-2, bias=0.0))

    def forward(self, input: Tensor):
        logits: Tensor = self.fc(input)
        logits = logits.reshape(*logits.shape[:-1], self.num_tokens, self.token_size)
        return D.Discrete(logits=logits)
