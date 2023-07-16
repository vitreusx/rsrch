from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import rsrch.distributions as D
import rsrch.distributions.constraints as C


class Normal(nn.Module):
    def __init__(self, in_features: int, out_shape: torch.Size | int):
        super().__init__()
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_shape = torch.Size(out_shape)
        self.out_features = int(np.prod(out_shape))
        self.fc = nn.Linear(in_features, 2 * self.out_features)

    def forward(self, x: Tensor) -> D.Distribution:
        params: Tensor = self.fc(x)
        mean, log_std = params.chunk(2, dim=1)  # [B, N_out]
        mean = mean.reshape(len(x), *self.out_shape)
        log_std = log_std.reshape(len(x), *self.out_shape)
        std = torch.exp(log_std)
        # This gets us a batch of normal distributions N(mean[i], std[i]^2 I)
        res_dist = D.Normal(mean, std, len(self.out_shape))
        return res_dist


class SquashedNormal(Normal):
    def __init__(self, in_features: int, min_values: Tensor, max_values: Tensor):
        super().__init__(in_features, min_values.shape)
        self.min_v, self.max_v = min_values, max_values
        self._loc: Tensor
        self.register_buffer("_loc", (self.min_v + self.max_v) / 2.0)
        self._scale: Tensor
        self.register_buffer("_scale", (self.max_v - self.min_v) / 2.0)

    def forward(self, x: Tensor) -> D.Distribution:
        normal: D.Distribution = super().forward(x)
        squash_fn = D.ComposeTransform(
            [
                D.SafeTanhTransform(cache_size=1),
                D.AffineTransform(self._loc, self._scale),
            ]
        )
        return D.TransformedDistribution(normal, squash_fn)


class Bernoulli(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> D.Distribution:
        logits: Tensor = self.net(x).ravel()
        return D.Bernoulli(logits=logits)


class Categorical(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> D.Categorical:
        logits: Tensor = self.fc(x)
        return D.Categorical(logits=logits)


class ToDistribution(nn.Module):
    def __init__(self, t: type, attr: str):
        super().__init__()
        self.t = t
        self.attr = attr

    def forward(self, x: Tensor):
        return self.t(**{self.attr: x})


class OHST(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> D.Categorical:
        logits: Tensor = self.fc(x)
        return D.OneHotCategoricalStraightThrough(logits=logits)


class MultiheadOHST(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> D.MultiheadOHST:
        logits: Tensor = self.net(x)
        return D.MultiheadOHST(self.out_features, self.num_classes, logits=logits)
