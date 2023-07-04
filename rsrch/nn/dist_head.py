import math

import numpy as np
import torch
import torch.distributions as D
import torch.distributions.constraints as C
import torch.nn as nn
from torch import Tensor


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
        res_dist = D.Independent(D.Normal(mean, std), 1)
        return res_dist


class SafeTanhTransform(D.Transform):
    domain = C.real
    codomain = C.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    eps = 1e-8
    min_v, max_v = -1.0 + eps, 1.0 - eps
    log2 = math.log(2)

    def __eq__(self, other):
        return isinstance(other, SafeTanhTransform)

    def _call(self, x: Tensor):
        return x.tanh()

    def _inverse(self, y: Tensor):
        y = y.clamp(self.min_v, self.max_v)
        return y.atanh()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor):
        return 2.0 * (self.log2 - x - nn.functional.softplus(-2.0 * x))


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
                SafeTanhTransform(cache_size=1),
                D.AffineTransform(self._loc, self._scale),
            ]
        )
        return D.TransformedDistribution(normal, squash_fn)


class Cast01Transform(D.Transform):
    domain = C.real
    codomain = C.boolean

    def __init__(self, prev_dt: torch.dtype):
        super().__init__()
        self.prev_dt = prev_dt

    def __eq__(self, other):
        return isinstance(other, Cast01Transform) and self.prev_dt == other.prev_dt

    def _call(self, x: Tensor):
        return x.bool()

    def _inverse(self, y: Tensor):
        return y.to(dtype=self.prev_dt)

    def log_abs_det_jacobian(self, x, y):
        return 1.0


class Bernoulli(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> D.Distribution:
        logits: Tensor = self.net(x).ravel()
        dist = D.Bernoulli(logits=logits)
        t = Cast01Transform(prev_dt=logits.dtype)
        return D.TransformedDistribution(dist, t)


class Categorical(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> D.Categorical:
        logits: Tensor = self.fc(x)
        return D.Categorical(logits=logits)
