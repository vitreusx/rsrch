import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces


def get_out_features(space: spaces.torch.Tensor):
    if isinstance(space, spaces.torch.Image):
        return space.num_channels
    else:
        return math.prod(space.shape)


def layer_init_(layer, std=math.sqrt(2), bias_const=0.0):
    if getattr(layer, "weight") is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if getattr(layer, "bias") is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Normal(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Tensor,
        std_type: Literal["const", "exp", "softplus"] = "softplus",
        min_std: float = 0.0,
    ):
        super().__init__()
        self.space = space
        self.norm_std = std_type
        self.min_std = min_std

        out_features = get_out_features(space)
        if std_type != "const":
            out_features *= 2

        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)

    def forward(self, input: Tensor):
        output: Tensor = self.layer(input)
        if self.norm_std != "const":
            output = output.reshape(-1, 2, *self.space.shape)
            mean, std = output.unbind(1)
            if self.norm_std == "exp":
                std = std.exp()
            elif self.norm_std == "softplus":
                std = F.softplus(std)
            if self.min_std > 0:
                std = self.min_std + std
        else:
            mean = output.reshape(-1, *self.space.shape)
            std = max(1.0, self.min_std)
        return D.Normal(mean, std, self._event_dims)


class MSEProxy(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
    ):
        super().__init__()
        self.space = space

        out_features = get_out_features(space)
        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)

    def forward(self, input: Tensor):
        value: Tensor = self.layer(input)
        value = value.reshape(-1, *self.space.shape)
        return D.MSEProxy(value, self._event_dims)


class Bernoulli(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Discrete,
    ):
        super().__init__()
        assert space.n == 2 and space.dtype == torch.bool
        self.layer = layer_ctor(1)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1)
        return D.Bernoulli(logits=logits)


class Categorical(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.space = space
        self.vocab_size = int(space.n)
        self.layer = layer_ctor(self.vocab_size)
        layer_init_(self.layer, std=1e-2)
        self._event_dims = len(space.shape)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.vocab_size)
        return D.Categorical(logits=logits, event_dims=self._event_dims)


class OneHot(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.OneHot,
    ):
        super().__init__()
        self.space = space
        self.vocab_size = int(space.shape[0])
        self.layer = layer_ctor(self.vocab_size)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.vocab_size)
        return D.OneHot(logits=logits)


class Discrete(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.TokenSeq,
    ):
        super().__init__()
        self.space = space
        self.num_tokens, self.vocab_size = space.num_tokens, space.vocab_size
        self.layer = layer_ctor(space.num_tokens * space.vocab_size)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.num_tokens, self.vocab_size)
        return D.Discrete(logits=logits)


class TruncNormal(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
        std_type: Literal["const", "exp", "softplus", "sigmoid2"] = "sigmoid2",
        min_std: float = math.exp(-5),
    ):
        super().__init__()
        self.space = space
        self.std_type = std_type
        self.min_std = min_std

        out_features = get_out_features(space)
        if std_type != "const":
            out_features *= 2

        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)
        self.register_buffer("loc", 0.5 * (space.low + space.high))
        self.register_buffer("scale", 0.5 * (space.high - space.low))

    def forward(self, input: Tensor):
        params: Tensor = self.layer(input)
        params = params.reshape(-1, 2, *self.space.shape)
        mean, std = params.unbind(1)

        mean = F.tanh(mean)

        if self.std_type == "exp":
            std = std.exp()
        elif self.std_type == "softplus":
            std = F.softplus(std)
        elif self.std_type == "sigmoid2":
            std = 2.0 * F.sigmoid(std / 2.0)

        if self.min_std > 0.0:
            std = std + self.min_std

        dist = D.Normal(mean, std, self._event_dims)
        dist = D.TruncNormal(dist, -1.0, 1.0)
        dist = D.Affine(dist, self.loc, self.scale)
        return dist


class TokenSeq(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.TokenSeq,
    ):
        super().__init__()
        self.space = space

        out_features = space.num_tokens * space.vocab_size
        self.layer = layer_ctor(out_features)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.space.num_tokens, self.space.vocab_size)
        return D.Categorical(logits=logits, event_dims=1)


DistType = Literal[
    "auto",
    "normal",
    "mse",
    "bern",
    "cat",
    "one_hot",
    "discrete",
    "trunc_normal",
    "token_seq",
]


def make(
    layer_ctor: Callable[[int], nn.Module],
    space: spaces.torch.Tensor,
    type: DistType = "auto",
    **kwargs,
):
    if type == "auto":
        if isinstance(space, spaces.torch.Discrete):
            type = "bern" if space.n <= 2 else "cat"
        elif isinstance(space, spaces.torch.Image):
            type = "mse"
        elif isinstance(space, spaces.torch.OneHot):
            type = "one_hot"
        elif isinstance(space, spaces.torch.Box):
            type = "trunc_normal" if space.bounded.all() else "normal"
        elif isinstance(space, spaces.torch.TokenSeq):
            type = "token_seq"

    cls = {
        "normal": Normal,
        "mse": MSEProxy,
        "bern": Bernoulli,
        "cat": Categorical,
        "one_hot": OneHot,
        "discrete": Discrete,
        "trunc_normal": TruncNormal,
        "token_seq": TokenSeq,
    }[type]

    return cls(layer_ctor, space, **kwargs.get(type, {}))


class OneHotWrapper(nn.Module):
    def __init__(self, dist_layer: nn.Module):
        super().__init__()
        self.dist_layer = dist_layer

    def forward(self, input: Tensor):
        dist: D.Categorical = self.dist_layer(input)
        if len(dist.event_shape) == 0:
            return D.OneHot(logits=dist.logits)
        else:
            return D.Discrete(logits=dist.logits)
