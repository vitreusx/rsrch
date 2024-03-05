from functools import partial
from typing import Literal

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
        out_shape: torch.Size | int,
        std: float | Literal["softplus", "sigmoid", "exp"] = "softplus",
    ):
        super().__init__()
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_shape = torch.Size(out_shape)
        self.out_features = int(np.prod(out_shape))
        self.std = std
        if isinstance(self.std, str):
            self.fc = nn.Linear(in_features, 2 * self.out_features)
        else:
            self.fc = nn.Linear(in_features, self.out_features)
        self.fc.apply(layer_init)

    def forward(self, x: Tensor) -> D.Distribution:
        params: Tensor = self.fc(x)
        if isinstance(self.std, str):
            mean, std = params.chunk(2, dim=1)  # [B, N_out]
            mean = mean.reshape(len(x), *self.out_shape)
            std = std.reshape(len(x), *self.out_shape)
            if self.std == "softplus":
                std = nn.functional.softplus(std)
            elif self.std == "sigmoid":
                std = std.sigmoid()
            elif self.std == "exp":
                std = std.exp()
        else:
            mean = params
            mean = mean.reshape(len(x), *self.out_shape)
            std = self.std

        # This gets us a batch of normal distributions N(mean[i], std[i]^2 I)
        res_dist = D.Normal(mean, std, event_dims=len(self.out_shape))
        return res_dist


class Bernoulli(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.fc.apply(layer_init)

    def forward(self, x: Tensor) -> D.Distribution:
        logits: Tensor = self.fc(x).ravel()
        return D.Bernoulli(logits=logits)


class Categorical(nn.Module):
    def __init__(self, in_features: int, num_classes: int, min_pr=None):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        self.fc.apply(layer_init)
        self._min_pr = min_pr

    def forward(self, x: Tensor) -> D.Categorical:
        if self._min_pr is None:
            logits: Tensor = self.fc(x)
            return D.Categorical(logits=logits)
        else:
            probs = nn.functional.softmax(self.fc(x), -1)
            probs = self._min_pr + (1.0 - self._min_pr) * probs
            return D.Categorical(probs=probs)


class OneHotCategoricalST(nn.Module):
    def __init__(self, in_features: int, num_classes: int, min_pr=None):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.fc.apply(layer_init)
        self._min_pr = min_pr

    def forward(self, x: Tensor) -> D.Categorical:
        if self._min_pr is None:
            logits: Tensor = self.fc(x)
            return D.OneHotCategoricalST(logits=logits)
        else:
            probs = nn.functional.softmax(self.fc(x), -1)
            probs = self._min_pr + (1.0 - self._min_pr) * probs
            return D.OneHotCategoricalST(probs=probs)


class MultiheadOHST(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_heads: int, min_pr=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.fc = nn.Linear(in_features, out_features)
        self.fc.apply(layer_init)
        self._min_pr = min_pr

    def forward(self, x: Tensor) -> D.MultiheadOHST:
        if self._min_pr is None:
            logits: Tensor = self.fc(x)
            return D.MultiheadOHST(self.num_heads, logits=logits)
        else:
            probs = nn.functional.softmax(self.fc(x), -1)
            probs = self._min_pr + (1.0 - self._min_pr) * probs
            return D.MultiheadOHST(self.num_heads, probs=probs)


class Dirac(nn.Module):
    def __init__(self, in_features: int, out_shape: torch.Size):
        super().__init__()
        self._out_shape = out_shape
        self.fc = nn.Linear(in_features, int(np.prod(out_shape)))
        self.fc.apply(layer_init)

    def forward(self, x: Tensor) -> D.Dirac:
        out = self.fc(x).reshape(-1, *self._out_shape)
        return D.Dirac(out, len(self._out_shape))


class TruncNormal(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_space: spaces.torch.Box,
    ):
        super().__init__()
        self._out_shape = act_space.shape
        self.register_buffer("low", act_space.low.cpu())
        self.register_buffer("high", act_space.high.cpu())
        # self.register_buffer("_loc", 0.5 * (act_space.low + act_space.high))
        # self.register_buffer("_scale", 0.5 * (act_space.high - act_space.low))
        # out_features = 2 * int(np.prod(act_space.shape))
        # self.fc = nn.Linear(in_features, out_features)
        # self.fc.apply(layer_init)
        act_dim = int(np.prod(act_space.shape))
        self.loc_fc = nn.Linear(in_features, act_dim)
        self.loc_fc.apply(layer_init)
        self.log_std = nn.Parameter(torch.zeros(1, *self._out_shape))

    def forward(self, x: Tensor):
        # out: Tensor = self.fc(x)
        # loc, scale = out.chunk(2, -1)
        # loc = loc.tanh().reshape(-1, *self._out_shape)
        # scale = F.softplus(scale.reshape(-1, *self._out_shape))
        # rv = D.Normal(loc, scale, len(self._out_shape))
        # rv = D.TruncNormal(rv, self.low, self.high)
        # rv = D.Affine(rv, self._loc, self._scale)
        loc: Tensor = self.loc_fc(x).reshape(-1, *self._out_shape)
        scale = F.softplus(self.log_std)
        rv = D.Normal(loc, scale, len(self._out_shape))
        rv = D.TruncNormal(rv, self.low, self.high)
        return rv


class Beta(nn.Module):
    def __init__(self, in_features: int, out_space: spaces.torch.Box):
        super().__init__()
        self.shape = out_space.shape
        out_dim = int(np.prod(out_space.shape))
        self.fc = nn.Linear(in_features, 2 * out_dim)
        self.fc.apply(partial(layer_init, std=1e-2))
        self.register_buffer("loc", out_space.low)
        self.register_buffer("scale", out_space.high - out_space.low)

    def forward(self, x: Tensor):
        out: Tensor = self.fc(x)
        alpha, beta = out.chunk(2, -1)
        alpha, beta = 1.0 + (alpha - 5.0).exp(), 1.0 + (beta - 5.0).exp()
        alpha, beta = alpha.reshape(-1, *self.shape), beta.reshape(-1, *self.shape)
        return D.Affine(
            D.Beta(alpha, beta, len(self.shape)),
            self.loc,
            self.scale,
        )
