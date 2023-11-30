from numbers import Number
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import rsrch.distributions as D


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


class SquashedNormal(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        low: Tensor | Number,
        high: Tensor | Number,
    ):
        super().__init__()
        self._out_shape = out_shape
        self.fc = nn.Linear(in_features, 2 * int(np.prod(out_shape)))
        self.fc.apply(layer_init)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x: Tensor) -> D.SquashedNormal:
        out = self.fc(x)
        mean, log_std = out.chunk(2, -1)
        mean = mean.reshape(-1, *self._out_shape)
        log_std = log_std.reshape(-1, *self._out_shape)
        std = log_std.exp()
        return D.SquashedNormal(mean, std, self.low, self.high, len(self._out_shape))


class TruncNormal(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        low: Tensor | Number,
        high: Tensor | Number,
    ):
        super().__init__()
        self._out_shape = out_shape
        self.fc = nn.Linear(in_features, 2 * int(np.prod(out_shape)))
        self.fc.apply(layer_init)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x: Tensor) -> D.SquashedNormal:
        out = self.fc(x)
        mean, std = out.chunk(2, -1)
        mean = mean.reshape(-1, *self._out_shape)
        std = nn.functional.softplus(std).reshape(-1, *self._out_shape)
        return D.TruncNormal(
            D.Normal(mean, std, len(self._out_shape)),
            self.low,
            self.high,
        )


class Piecewise(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        num_buckets: int,
        low: Tensor | Number,
        high: Tensor | Number,
    ):
        super().__init__()
        self._out_shape = out_shape
        self._num_buckets = num_buckets
        out_features = num_buckets * int(np.prod(out_shape))
        self.fc = nn.Linear(in_features, out_features)
        self.fc.apply(layer_init)
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, x: Tensor) -> D.SquashedNormal:
        logits = self.fc(x)
        logits = logits.reshape(-1, *self._out_shape, self._num_buckets)
        return D.Piecewise(
            D.Categorical(logits=logits),
            self.low,
            self.high,
            len(self._out_shape),
        )
