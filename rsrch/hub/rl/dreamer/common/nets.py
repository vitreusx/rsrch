import math
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from . import dh
from .utils import to_camel_case

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
            self._norm = nn.LayerNorm(3 * hidden_size, eps=1e-3)
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


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        return x.reshape(*x.shape[:1], -1)


class BoxEncoder(nn.Sequential):
    def __init__(self, space: spaces.torch.Box, **mlp):
        in_features = math.prod(space.shape)
        super().__init__(
            Flatten(),
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


class DictEncoder(nn.ModuleDict):
    def __init__(
        self,
        space: spaces.torch.Dict,
        encoders: dict[str, nn.Module],
        keys=None,
    ):
        if keys is not None:
            encoders = {key: encoders[key] for key in keys}
        super().__init__(encoders)
        self.space = space

    def forward(self, input: dict):
        outputs = [self[key](input[key]) for key in self]
        return torch.cat(outputs, dim=1)


def AutoEncoder(space: spaces.torch.Tensor, **args):
    if isinstance(space, spaces.torch.Dict):
        return DictEncoder(
            space,
            {key: AutoEncoder(value, **args) for key, value in space.items()},
            **args.get("dict", {}),
        )
    elif isinstance(space, spaces.torch.Image):
        return ImageEncoder(space, **args.get("image", {}))
    elif isinstance(space, spaces.torch.Discrete):
        return DiscreteEncoder(space, **args.get("discrete", {}))
    elif isinstance(space, spaces.torch.Box):
        return BoxEncoder(space, **args.get("box", {}))
    else:
        raise ValueError(type(space))


def make_encoder(space: spaces.torch.Tensor, **kwargs):
    cls_type = kwargs["type"]
    cls = globals()[to_camel_case(f"{cls_type}_encoder")]
    del kwargs["type"]
    if cls_type == "auto":
        return cls(space, **kwargs)
    else:
        return cls(space, **kwargs.get(cls_type, {}))


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


class ConstDecoder(nn.Module):
    def __init__(self, in_features: int, space, value):
        super().__init__()
        self.in_features = in_features
        self.space = space
        self.value: Tensor
        self.register_buffer("value", torch.as_tensor(value))

    def forward(self, input: Tensor):
        value = self.value.expand(input.shape[0], *self.value.shape)
        return D.Dirac(value, len(self.value.shape))


class DictDecoder(nn.ModuleDict):
    def __init__(
        self,
        in_features: int,
        space: spaces.torch.Dict,
        decoders: dict[str, nn.Module],
        keys: list[str] | None = None,
    ):
        if keys is not None:
            decoders = {key: decoders[key] for key in keys}
        super().__init__(decoders)
        self.in_features = in_features
        self.space = space

    def forward(self, input: Tensor):
        return {name: self[name](input) for name in self}


def AutoDecoder(in_features: int, space: spaces.torch.Tensor, **args):
    if isinstance(space, spaces.torch.Dict):
        return DictDecoder(
            in_features,
            space,
            {
                name: AutoDecoder(in_features, value, **args)
                for name, value in space.items()
            },
            **args.get("dict", {}),
        )
    elif isinstance(space, spaces.torch.Image):
        return ImageDecoder(in_features, space, **args.get("image", {}))
    elif isinstance(space, spaces.torch.Discrete):
        return DiscreteDecoder(in_features, space, **args.get("discrete", {}))
    elif isinstance(space, spaces.torch.Box):
        return BoxDecoder(in_features, space, **args.get("box", {}))
    else:
        raise ValueError(type(space))


def make_decoder(in_features: int, space: spaces.torch.Tensor, **kwargs):
    cls_type = kwargs["type"]
    cls = globals()[to_camel_case(f"{cls_type}_decoder")]
    del kwargs["type"]
    if cls_type == "auto":
        return cls(in_features, space, **kwargs)
    else:
        return cls(in_features, space, **kwargs.get(cls_type, {}))


class ActionEncoder(nn.Module):
    """An action encoder. We cannot use a regular encoder, because:
    - the output of the actor needs to be in the same format as the
      output of the action encoder, in particular differentiable
    - we need to be able to get the un-encoded action from this output, for use
      in the env interaction."""

    def __init__(self, act_space: spaces.torch.Tensor):
        super().__init__()
        self.act_space = act_space

    def forward(self, act: Tensor):
        if isinstance(self.act_space, spaces.torch.Discrete):
            act = F.one_hot(act, self.act_space.n)
        elif isinstance(self.act_space, spaces.torch.TokenSeq):
            act = F.one_hot(act, self.act_space.vocab_size)
        return act.flatten(1).float()

    def inverse(self, act: Tensor):
        if isinstance(self.act_space, spaces.torch.Discrete):
            act = act.argmax(-1)
        elif isinstance(self.act_space, spaces.torch.TokenSeq):
            act = act.reshape(-1, self.act_space.num_tokens, self.act_space.vocab_size)
            act = act.argmax(-1)
        else:
            act = act.reshape(-1, *self.act_space.shape)
        return act


def ActorHead(in_features: int, act_space: spaces.torch.Tensor, **kwargs):
    layer_ctor = partial(nn.Linear, in_features)
    dist_layer = dh.make(layer_ctor, act_space, **kwargs)
    if not act_space.dtype.is_floating_point:
        dist_layer = dh.OneHotWrapper(dist_layer)
    return dist_layer


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
