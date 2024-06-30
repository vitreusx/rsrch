from functools import partial
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Identity

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import *
from rsrch.nn import dist_head as dh
from rsrch.nn.utils import pass_gradient, safe_mode
from rsrch.types import Tensorlike

from . import config


def ActLayer(type: config.ActType) -> Callable[[], nn.Module]:
    return {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[type]


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


def NormLayer1d(type: config.NormType) -> Callable[[int], nn.Module]:
    return {
        "none": lambda _: nn.Identity(),
        "batch": nn.BatchNorm1d,
        "layer": nn.LayerNorm,
    }[type]


def NormLayer2d(type: config.NormType) -> Callable[[int], nn.Module]:
    return {
        "none": lambda _: nn.Identity(),
        "batch": nn.BatchNorm2d,
        "layer": LayerNorm2d,
    }[type]


class GRUCellLN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        act="tanh",
        norm=False,
        update_bias: float = -1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act = act
        self.norm = norm
        self.update_bias = update_bias

        self._layer = nn.Linear(
            input_size + hidden_size,
            3 * hidden_size,
            bias=not norm,
        )

        if self.norm:
            self._norm = nn.LayerNorm(3 * hidden_size)
        self._act = ActLayer(act)()

    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        parts: Tensor = self._layer(torch.cat((input, hidden), -1))
        if self.norm:
            parts = self._norm(parts)
        reset, cand, update = parts.chunk(3, -1)
        cand = self._act(reset.sigmoid() * cand)
        update = (update + self.update_bias).sigmoid()
        out = update * cand + (1 - update) * hidden
        return out


class ImageEncoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Image,
        depth: int = 48,
        kernels: list[int] = [4, 4, 4, 4],
        norm: config.NormType = "none",
        act: config.ActType = "elu",
    ):
        norm_layer = NormLayer1d(type=norm)
        act_layer = ActLayer(act)

        channels = [
            space.num_channels,
            *(2**i * depth for i in range(len(kernels))),
        ]

        layers = []
        for in_channels, out_channels, kernel in zip(channels, channels[1:], kernels):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel, 2),
                norm_layer(out_channels),
                act_layer(),
            ]

        layers += [nn.Flatten(1)]
        super().__init__(*layers)
        self.space = space


class ImageDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Image,
        depth: int = 48,
        kernels: list[int] = [5, 5, 6, 6],
        norm: config.NormType = "none",
        act: config.ActType = "elu",
    ):
        norm_layer = NormLayer2d(type=norm)
        act_layer = ActLayer(act)

        channels = [32 * depth]
        for i in reversed(range(len(kernels) - 1)):
            channels += [2**i * depth]
        channels += [space.num_channels]

        layers = [
            nn.Linear(in_features, 32 * depth),
            Reshape(-1, (32 * depth, 1, 1)),
        ]
        for in_channels, out_channels, kernel in zip(channels, channels[1:], kernels):
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel, 2),
                norm_layer(out_channels),
                act_layer(),
            ]

        super().__init__(*layers)

    def forward(self, input: Tensor):
        out = super().forward(input)
        return D.MSEProxy(out, 3)


class FC(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int | None,
        hidden: int,
        layers: int,
        norm: config.NormType = "none",
        act: config.ActType = "elu",
    ):
        norm_layer = NormLayer1d(type=norm)
        act_layer = ActLayer(act)

        layer_sizes = [
            in_features,
            *[hidden for _ in range(layers)],
            *([out_features] if out_features is not None else []),
        ]

        layers = []
        for in_features_, out_features_ in zip(layer_sizes, layer_sizes[1:]):
            layers += [
                nn.Linear(in_features_, out_features_),
                norm_layer(out_features_),
                act_layer(),
            ]

        if out_features is not None:
            layers = layers[:-2]

        super().__init__(*layers)
        self.in_features = in_features
        self.out_features = out_features or hidden
        self.hidden = hidden
        self.layers = layers


class BoxEncoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Box,
        **fc_args,
    ):
        in_features = int(np.prod(space.shape))
        super().__init__(
            nn.Flatten(1),
            FC(in_features, None, **fc_args),
        )
        self.space = space


class BoxDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Box,
        dist: Literal["mse", "auto"] = "auto",
        **fc_args,
    ):
        super().__init__()
        self.in_features = in_features
        self.space = space
        self.dist = dist

        if dist == "mse":
            out_features = int(np.prod(space.shape))
            self.fc = FC(in_features, out_features, **fc_args)
        elif dist == "auto":
            self.fc = FC(in_features, None, **fc_args)
            if space.dtype == torch.bool:
                self.head = dh.Bernoulli(self.fc.out_features)
            elif space.bounded.all():
                self.head = dh.Beta(self.fc.out_features, space)
            else:
                self.head = dh.Normal(
                    self.fc.out_features, space.shape, std_act="softplus"
                )

    def forward(self, input: Tensor):
        out = self.fc(input)
        if self.dist == "mse":
            out = out.reshape(-1, *self.space.shape)
            return D.MSEProxy(out, len(self.space.shape))
        else:
            return self.head(out)


class DiscreteEncoder(nn.Module):
    def __init__(self, space: spaces.torch.Discrete):
        super().__init__()
        self.space = space
        self.n = space.n

    def forward(self, input: Tensor):
        return F.one_hot(input, self.n).float()

    def inverse(self, input: Tensor):
        return input.argmax(-1)


class DiscreteDecoder(nn.Sequential):
    def __init__(self, in_features: int, space: spaces.torch.Discrete, **fc_args):
        fc = FC(in_features, None, **fc_args)
        if space.dtype == torch.bool:
            head = dh.Bernoulli(fc.out_features)
        else:
            head = dh.Categorical(fc.out_features, space.n)

        super().__init__(fc, head)


def AutoEncoder(space: spaces.torch.Space, **args):
    if type(space) == spaces.torch.Image:
        return ImageEncoder(space, **args.get("image", {}))
    elif type(space) == spaces.torch.Box:
        return BoxEncoder(space, **args.get("box", {}))
    elif type(space) == spaces.torch.Discrete:
        return DiscreteEncoder(space, **args.get("discrete", {}))


def AutoDecoder(in_features: int, space: spaces.torch.Space, **args):
    if type(space) == spaces.torch.Image:
        return ImageDecoder(in_features, space, **args.get("image", {}))
    elif type(space) == spaces.torch.Box:
        return BoxDecoder(in_features, space, **args.get("box", {}))
    elif type(space) == spaces.torch.Discrete:
        return DiscreteDecoder(in_features, space, **args.get("discrete", {}))


class StreamNorm(nn.Module):
    def __init__(
        self,
        shape=(),
        momentum=0.99,
        scale=1.0,
        eps=1e-8,
        affine=False,
    ):
        super().__init__()
        self.shape = shape
        self.momentum = momentum
        self.scale = scale
        self.eps = eps
        self.affine = affine

        self._mag: Tensor
        self.register_buffer("_mag", torch.ones(shape, dtype=torch.float64))

    def forward(self, input: Tensor):
        self._update(input)
        input = input / (self._mag.type_as(input) + self.eps)
        input = input * self.scale
        return input

    def _reset(self):
        self._mag.fill_(1.0)

    def _update(self, input: Tensor):
        input = input.reshape(-1, *self.shape)
        mag = input.abs().mean(0).to(torch.float64)
        value = self.momentum * self._mag + (1.0 - self.momentum) * mag
        self._mag.copy_(value)


class PolicyLayer(nn.Module):
    def __init__(self, in_features: int, act_enc: nn.Module):
        super().__init__()
        self.in_features = in_features
        self.act_enc = act_enc
        self.space = act_enc.space

        if isinstance(act_enc, DiscreteEncoder):
            self.space: spaces.torch.Discrete
            self._fc = nn.Linear(in_features, act_enc.n)
        elif isinstance(act_enc, BoxEncoder):
            self.space: spaces.torch.Box
            self._head = dh.Beta(in_features, act_enc.space)

    def forward(self, state: Tensor) -> D.Distribution:
        if isinstance(self.act_enc, DiscreteEncoder):
            logits = self._fc(state)
            dist = D.OneHot(logits=logits)
        elif isinstance(self.act_enc, BoxEncoder):
            dist = self._head(state)
        return dist
