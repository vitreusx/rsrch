import math
from copy import copy
from typing import Callable, Iterable, ParamSpec, TypeVar, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization

from rsrch import spaces


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
    def __init__(self, shape: tuple[int, ...] = ()):
        super().__init__()
        self.shape = shape

        self._first_time = True
        self.loc: Tensor
        self.register_buffer("loc", torch.zeros(shape))
        self.inv_scale: Tensor
        self.register_buffer("inv_scale", torch.ones(shape))

    def forward(self, input: Tensor):
        if self.training and self._first_time:
            mean = input.reshape(-1, *self.shape).mean(0)
            self.loc.copy_(mean)
            std = input.reshape(-1, *self.shape).std(0)
            self.inv_scale.copy_(std.clamp_min(1e-8).reciprocal())
            self._first_time = False
        input = (input - self.loc) * self.inv_scale
        return input

    def inv(self, input: Tensor):
        return input / self.inv_scale + self.loc


class LayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
    ):
        super().__init__()
        self._norm = nn.LayerNorm(
            normalized_shape=(num_features,),
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(self, input: Tensor):
        input = input.moveaxis(-3, -1)
        input = self._norm(input)
        input = input.moveaxis(-1, -3)
        return input


class ResBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        norm_layer: Callable[[int], nn.Module] | None,
        act_layer: Callable[[], nn.Module],
    ):
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
        *,
        conv_hidden: int = 32,
        scale_factor: int = 16,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm2d,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
        flatten: bool = True,
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

        if flatten:
            layers.append(nn.Flatten())

        super().__init__(*layers)


class Decoder_v1(nn.Module):
    def __init__(
        self,
        in_features: int | None,
        obs_space: spaces.torch.Image,
        *,
        conv_hidden: int = 48,
        scale_factor: int = 32,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm2d,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
    ):
        super().__init__()

        assert obs_space.width % scale_factor == 0
        in_width = obs_space.width // scale_factor

        assert obs_space.height % scale_factor == 0
        in_height = obs_space.height // scale_factor

        in_channels = conv_hidden * (scale_factor // 2)

        layers = []

        in_shape = (in_channels, in_height, in_width)
        if in_features is not None:
            layers += [
                nn.Linear(in_features, int(np.prod(in_shape))),
                act_layer(),
                Reshape(-1, in_shape),
            ]

        num_stages = int(math.log2(scale_factor))
        channels = [2**stage * conv_hidden for stage in reversed(range(num_stages))]
        channels += [obs_space.num_channels]

        for _ in range(res_blocks):
            layers += [ResBlock(channels[0], norm_layer, act_layer)]

        for in_channels, out_channels in zip(channels, channels[1:]):
            layers += [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)]
            if norm_layer is not None:
                layers += [norm_layer(out_channels)]
            layers += [act_layer()]

        while not isinstance(layers[-1], nn.ConvTranspose2d):
            layers.pop()

        self.main = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        return self.main(input)


class DreamerEncoder(nn.Sequential):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        *,
        depth=48,
        act_layer=nn.ELU,
        norm_layer=None,
    ):
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [
            nn.Conv2d(obs_space.num_channels, depth, 4, 2),
            norm_layer(depth),
            act_layer(),
            nn.Conv2d(depth, 2 * depth, 4, 2),
            norm_layer(2 * depth),
            act_layer(),
            nn.Conv2d(2 * depth, 4 * depth, 4, 2),
            norm_layer(4 * depth),
            act_layer(),
            nn.Conv2d(4 * depth, 8 * depth, 4, 2),
            norm_layer(8 * depth),
            act_layer(),
            nn.Flatten(),
        ]

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        super().__init__(*layers)


class DreamerDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        obs_space: spaces.torch.Image,
        *,
        depth=48,
        act_layer=nn.ELU,
    ):
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [
            nn.Linear(in_features, 32 * depth),
            Reshape(32 * depth, (32 * depth, 1, 1)),
            nn.ConvTranspose2d(32 * depth, 4 * depth, 5, 2),
            norm_layer(4 * depth),
            act_layer(),
            nn.ConvTranspose2d(4 * depth, 2 * depth, 5, 2),
            norm_layer(2 * depth),
            act_layer(),
            nn.ConvTranspose2d(2 * depth, depth, 6, 2),
            norm_layer(depth),
            act_layer(),
            nn.ConvTranspose2d(depth, obs_space.num_channels, 6, 2),
        ]

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        super().__init__(*layers)
