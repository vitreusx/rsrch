from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import rsrch.distributions as D


class Normal(nn.Module):
    def __init__(self, in_features: int, out_shape: torch.Size | int, std="softplus"):
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


class OneHotCategoricalST(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> D.Categorical:
        logits: Tensor = self.fc(x)
        return D.OneHotCategoricalST(logits=logits)


class MultiheadOHST(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> D.MultiheadOHST:
        logits: Tensor = self.net(x)
        return D.MultiheadOHST(self.num_heads, logits=logits)
