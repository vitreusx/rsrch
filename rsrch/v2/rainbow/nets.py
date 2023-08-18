import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D

from . import config
from .distq import ValueDist


class NatureEncoder(nn.Sequential):
    def __init__(self, obs_shape: torch.Size):
        in_channels, height, width = obs_shape
        assert (height, width) == (84, 84)
        self.out_features = 3136

        super().__init__(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )


class ImpalaSmall(nn.Sequential):
    def __init__(self, obs_shape: torch.Size):
        super().__init__(
            nn.Conv2d(obs_shape[0], 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
        )
        self.out_features = 1152


class ImpalaResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ImpalaResidual(out_channels),
            ImpalaResidual(out_channels),
        )


class ImpalaLarge(nn.Sequential):
    def __init__(self, obs_shape: torch.Size, model_size=1):
        super().__init__(
            ImpalaBlock(obs_shape[0], 16 * model_size),
            ImpalaBlock(16 * model_size, 32 * model_size),
            ImpalaBlock(32 * model_size, 64 * model_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten(),
        )
        self.out_features = 4096 * model_size


class QHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_actions: int,
        dist: config.Dist,
        hidden=256,
    ):
        super().__init__()
        self._num_actions = num_actions
        self._dist = dist

        self.value_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        if self._dist.enabled:
            adv_out = num_actions * self._dist.num_atoms
        else:
            adv_out = num_actions

        self.adv_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, adv_out),
        )

    def forward(self, feat: Tensor) -> Tensor | ValueDist:
        value_out, adv_out = self.value_stream(feat), self.adv_stream(feat)

        if self._dist.enabled:
            value_out = value_out.reshape(-1, self._num_actions, 1)
            adv_out = adv_out.reshape(-1, self._num_actions, self._dist.num_atoms)
            logits = value_out + adv_out - adv_out.mean(-2, keepdim=True)
            return ValueDist(
                v_min=self._dist.v_min,
                v_max=self._dist.v_max,
                N=self._dist.num_atoms,
                index_rv=D.Categorical(logits=logits),
            )
        else:
            return value_out + adv_out - adv_out.mean(-1, keepdim=True)


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma0: float,
        bias=True,
        factorized=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._sigma0 = sigma0
        self._bias = bias
        self._factorized = factorized

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.empty_like(self.weight))
        self.register_buffer("weight_eps", torch.empty_like(self.weight))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.noisy_bias = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_eps", torch.empty_like(self.bias))

        self.reset_weights()
        self.reset_noise()

    def reset_weights(self):
        s = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -s, s)
        nn.init.constant_(self.weight_eps, self._sigma0 * s)
        if self._bias:
            nn.init.uniform_(self.bias, -s, s)
            nn.init.constant_(self.bias_eps, self._sigma0 * s)

    def reset_noise(self):
        device, dtype = self.weight.device, self.weight.dtype

        eps_in = torch.randn(self.in_features, device=device, dtype=dtype)
        sign_in = self.eps_in.sign()
        eps_in.abs_().sqrt_().mul_(sign_in)

        eps_out = torch.randn(self.out_features, device=device, dtype=dtype)
        sign_out = self.weight_eps_v.sign()
        eps_out.abs_().sqrt_().mul_(sign_out)

        self.weight_eps.copy_(eps_out.outer(eps_in))
        if self._bias:
            self.bias_eps.copy_(eps_out)

    def forward(self, x):
        w = self.weight + self.noisy_weight * self.weight_eps
        if self._bias:
            b = self.bias + self.noisy_bias * self.bias_eps
            return F.linear(x, w, b)
        else:
            return F.linear(x, w)
