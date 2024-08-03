import math
from functools import partial
from typing import Callable, Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from . import dh

ActType = Literal["relu", "elu", "tanh"]
ACT_LAYERS = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}


def ActLayer(type: ActType) -> nn.Module:
    return ACT_LAYERS[type]()


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super().__init__()
        self._norm = nn.LayerNorm(normalized_shape=(num_features,), **kwargs)

    def forward(self, input: Tensor):
        input = input.moveaxis(-3, -1)
        input = self._norm(input)
        input = input.moveaxis(-1, -3)
        return input


NormType = Literal["none", "batch", "layer"]
NORM_LAYERS_1D = {
    "none": lambda _: nn.Identity(),
    "batch": nn.BatchNorm1d,
    "layer": nn.LayerNorm,
}
NORM_LAYERS_2D = {
    "none": lambda _: nn.Identity(),
    "batch": nn.BatchNorm2d,
    "layer": LayerNorm2d,
}


def NormLayer1d(type: NormType, in_features: int) -> nn.Module:
    return NORM_LAYERS_1D[type](in_features)


def NormLayer2d(type: NormType, in_channels: int) -> nn.Module:
    return NORM_LAYERS_2D[type](in_channels)


class Reshape(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        if isinstance(in_shape, int):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        if isinstance(out_shape, int):
            out_shape = (out_shape,)
        self.out_shape = tuple(out_shape)

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class GRUCellLN(nn.Module):
    __constants__ = ["input_size", "hidden_size", "update_bias", "norm"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        act: ActLayer = "tanh",
        norm: bool = False,
        update_bias: float = -1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act = act
        self.norm = norm
        self.update_bias = update_bias

        self._layer = nn.Linear(
            input_size + hidden_size, 3 * hidden_size, bias=not norm
        )

        if self.norm:
            self._norm = nn.LayerNorm(3 * hidden_size)
        self._act = ActLayer(act)

    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        parts: Tensor = self._layer(torch.cat((input, hidden), -1))
        if self.norm:
            parts = self._norm(parts)
        reset, cand, update = parts.chunk(3, -1)
        cand = self._act(reset.sigmoid() * cand)
        update = (update + self.update_bias).sigmoid()
        out = update * cand + (1 - update) * hidden
        return out


class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int | None,
        hidden: int,
        layers: int,
        norm: NormType = "none",
        act: ActType = "elu",
    ):
        norm_layer = partial(NormLayer1d, norm)
        act_layer = partial(ActLayer, act)

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


class ImageEncoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Image,
        depth: int = 48,
        kernels: list[int] = [4, 4, 4, 4],
        norm: NormType = "none",
        act: ActType = "elu",
    ):
        norm_layer = partial(NormLayer1d, norm)
        act_layer = partial(ActLayer, act)

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

    def forward(self, input: Tensor):
        input = input - 0.5
        return super().forward(input)


class BoxEncoder(nn.Sequential):
    def __init__(self, space: spaces.torch.Box, **mlp):
        in_features = math.prod(space.shape)
        super().__init__(
            nn.Flatten(1),
            MLP(in_features, None, **mlp),
        )
        self.space = space


class DiscreteEncoder(nn.Module):
    def __init__(self, space: spaces.torch.Discrete):
        super().__init__()
        self.space = space
        self.n = space.n

    def forward(self, input: Tensor):
        return F.one_hot(input, self.n).float()

    def inverse(self, input: Tensor):
        return input.argmax(-1)


def AutoEncoder(space: spaces.torch.Space, **args):
    if type(space) == spaces.torch.Image:
        return ImageEncoder(space, **args.get("image", {}))
    elif type(space) == spaces.torch.Box:
        return BoxEncoder(space, **args.get("box", {}))
    elif type(space) == spaces.torch.Discrete:
        return DiscreteEncoder(space, **args.get("discrete", {}))


class ImageDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Image,
        depth: int = 48,
        kernels: list[int] = [5, 5, 6, 6],
        norm: NormType = "none",
        act: ActType = "elu",
        dist={},
    ):
        norm_layer = partial(NormLayer2d, norm)
        act_layer = partial(ActLayer, act)

        channels = [32 * depth]
        for i in reversed(range(len(kernels) - 1)):
            channels += [2**i * depth]

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

        layer_ctor = lambda out: nn.ConvTranspose2d(channels[-1], out, kernels[-1], 2)
        layers.append(dh.make(layer_ctor, space, **dist))

        super().__init__(*layers)

    def forward(self, input: Tensor):
        dist = super().forward(input)
        return D.Affine(dist, loc=0.5, scale=1.0)


class BoxDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Box,
        hidden: int,
        layers: int,
        norm: NormType = "none",
        act: ActType = "elu",
        dist={},
    ):
        body = MLP(in_features, None, hidden, layers, norm, act)
        layer_ctor = lambda out: nn.Linear(body.out_features, out)
        head = dh.make(layer_ctor, space, **dist)
        super().__init__(body, head)


class DiscreteDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Discrete,
        **mlp,
    ):
        mlp = MLP(in_features, None, **mlp)
        layer_ctor = lambda out: nn.Linear(mlp.out_features, out)
        head = dh.make(layer_ctor, space)
        super().__init__(mlp, head)


def AutoDecoder(in_features: int, space: spaces.torch.Space, **args):
    if type(space) == spaces.torch.Image:
        return ImageDecoder(in_features, space, **args.get("image", {}))
    elif type(space) == spaces.torch.Box:
        return BoxDecoder(in_features, space, **args.get("box", {}))
    elif type(space) == spaces.torch.Discrete:
        return DiscreteDecoder(in_features, space, **args.get("discrete", {}))


class ActionEncoder(nn.Module):
    """An action encoder. We cannot use a regular encoder, because:
    - the output of the actor needs to be in the same format as the
      output of the action encoder, in particular differentiable
    - we need to be able to get the un-encoded action from this output, for use
      in the env interaction."""

    def __init__(self, act_space: spaces.torch.Space):
        super().__init__()
        assert isinstance(act_space, (spaces.torch.Discrete, spaces.torch.Box))
        self.act_space = act_space

    def forward(self, act: Tensor):
        if isinstance(self.act_space, spaces.torch.Discrete):
            return F.one_hot(act, self.act_space.n).float()
        else:
            return act.flatten(1)

    def inverse(self, act: Tensor):
        if isinstance(self.act_space, spaces.torch.Discrete):
            return act.argmax(-1)
        else:
            return act.reshape(-1, *self.act_space.shape)


def ActorHead(in_features: int, act_space: spaces.torch.Space):
    layer_ctor = partial(nn.Linear, in_features)
    if isinstance(act_space, spaces.torch.Discrete):
        return dh.OneHot(layer_ctor, act_space)
    elif isinstance(act_space, spaces.torch.Box):
        return dh.make(layer_ctor, act_space, type="auto")


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
