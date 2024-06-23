import math
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import safe_mode

from .utils import over_seq
from .vq import *


def Sequential(*layers):
    layers = [x for x in layers if not isinstance(x, nn.Identity)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, num_features: int, norm_layer, act_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            *([norm_layer(num_features)] if norm_layer else []),
            act_layer(),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            *([norm_layer(num_features)] if norm_layer else []),
        )
        self.act = act_layer()

    def forward(self, input: Tensor):
        return self.act(input + self.net(input))


class Encoder_v1(nn.Sequential):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        conv_hidden: int,
        *,
        scale_factor: int = 16,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
    ):
        assert obs_space.width % scale_factor == 0
        assert obs_space.height % scale_factor == 0

        num_stages = int(math.log2(scale_factor))
        assert 2**num_stages == scale_factor
        channels = [obs_space.num_channels]
        channels.extend(2**stage * conv_hidden for stage in range(num_stages))

        layers = []
        for idx, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            if idx > 0:
                if norm_layer is not None:
                    layers.append(norm_layer(in_ch))
                layers.append(act_layer())
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1))

        for _ in range(res_blocks):
            layers.append(ResBlock(channels[-1], norm_layer, act_layer))

        super().__init__(*layers)


class Decoder_v1(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        conv_hidden: int,
        *,
        scale_factor: int = 16,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        assert obs_space.width % scale_factor == 0
        cur_w = obs_space.width // scale_factor

        assert obs_space.height % scale_factor == 0
        cur_h = obs_space.height // scale_factor

        cur_nc = conv_hidden * (scale_factor // 2)

        self.in_shape = (cur_nc, cur_h, cur_w)

        layers = [
            ResBlock(conv_hidden, norm_layer, act_layer),
            ResBlock(conv_hidden, norm_layer, act_layer),
        ]

        num_stages = int(math.log2(scale_factor))
        assert scale_factor == 2**num_stages
        channels = [2**stage * conv_hidden for stage in reversed(range(num_stages))]
        channels += [obs_space.num_channels]

        layers = []
        for _ in range(res_blocks):
            layers.append(ResBlock(channels[0], norm_layer, act_layer))

        for stage, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
            if stage < num_stages - 1:
                if norm_layer is not None:
                    layers.append(norm_layer(out_ch))
                layers.append(act_layer())

        self.main = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        return self.main(input)


class Reshape(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        if not isinstance(in_shape, Iterable):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        if not isinstance(out_shape, Iterable):
            out_shape = (out_shape,)
        self.out_shape = tuple(out_shape)

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class Normalize(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.first_time = True
        self.loc: Tensor
        self.register_buffer("loc", torch.zeros([]))
        self.inv_scale: Tensor
        self.register_buffer("inv_scale", torch.ones([]))

    def forward(self, input: Tensor):
        if self.training and self.first_time:
            self.loc.copy_(input.mean())
            self.inv_scale.copy_(input.std().clamp_min(1e-8).reciprocal())
            self.first_time = False
        input = (input - self.loc) * self.inv_scale
        return input

    def inverse(self, input: Tensor):
        return input / self.inv_scale + self.loc


class AtariEncoder(nn.Sequential):
    def __init__(self, hidden=48, in_channels=1):
        super().__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(hidden, 2 * hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(2 * hidden, 4 * hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(4 * hidden, 8 * hidden, 4, 2),
            nn.ELU(),
            nn.Flatten(),
        )


class Reshape(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        if not isinstance(in_shape, Iterable):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        if not isinstance(out_shape, Iterable):
            out_shape = (out_shape,)
        self.out_shape = tuple(out_shape)

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class Lipschitz(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = P.spectral_norm(net)
        # device = next(self.net.parameters()).device
        # log_K0 = np.log(np.e - 1)  # F.softplus(log_K0) = 1
        # self.log_K = nn.Parameter(log_K0 * torch.ones((), device=device))

    def forward(self, input: Tensor):
        return self.net(input)
        # return self.net(input) * F.softplus(self.log_K)


class AtariDecoder(nn.Sequential):
    def __init__(self, in_features: int, hidden=48, out_channels=1):
        layers = [
            nn.Linear(in_features, in_features),
            nn.ELU(),
            nn.Linear(in_features, 32 * hidden),
            Reshape(32 * hidden, (32 * hidden, 1, 1)),
            nn.ConvTranspose2d(32 * hidden, 4 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(4 * hidden, 2 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(2 * hidden, hidden, 6, 2),
            nn.ELU(),
            nn.ConvTranspose2d(hidden, out_channels, 6, 2),
        ]

        for idx in range(len(layers)):
            if isinstance(layers[idx], (nn.Linear, nn.ConvTranspose2d)):
                layers[idx] = Lipschitz(layers[idx])

        super().__init__(*layers)
