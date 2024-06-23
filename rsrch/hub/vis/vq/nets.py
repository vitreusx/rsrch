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
