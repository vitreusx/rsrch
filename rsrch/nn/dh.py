import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces

from .utils import tf_init_


def get_out_features(space: spaces.torch.Tensor):
    if isinstance(space, spaces.torch.Image):
        return space.num_channels
    else:
        return math.prod(space.shape)


def process_std(
    std: Tensor,
    std_type: Literal["const", "exp", "softplus"],
    init_std: float,
    min_std: float,
):
    if init_std > 0.0:
        std = std + init_std

    if std_type == "exp":
        std = std.exp()
    elif std_type == "softplus":
        std = F.softplus(std)
    elif std_type == "sigmoid2":
        std = 2.0 * F.sigmoid(std / 2.0)

    if min_std > 0.0:
        std = std + min_std

    return std


class Normal(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Tensor,
        std_type: Literal["const", "exp", "softplus"] = "softplus",
        std_value: float | None = None,
        init_std: float = 0.0,
        min_std: float = 0.1,
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        self.space = space
        self.std_type = std_type
        self.norm_std = std_type
        self.init_std = init_std
        self.min_std = min_std
        self.std_value = std_value

        out_features = get_out_features(space)
        self.mean_fc = layer_ctor(out_features)
        if self.std_type != "const":
            self.std_fc = layer_ctor(out_features)

        self._event_dims = len(space.shape)

        if init == "tf":
            self.apply(tf_init_)

    def forward(self, input: Tensor):
        mean: Tensor = self.mean_fc(input)
        mean = mean.reshape(-1, *self.space.shape)
        if self.std_type != "const":
            std: Tensor = self.std_fc(input)
            std = std.reshape(-1, *self.space.shape)
            std = process_std(std, self.std_type, self.init_std, self.min_std)
        else:
            std = self.std_value
        return D.Normal(mean, std, self._event_dims)


class TruncNormal(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
        std_type: Literal["exp", "softplus", "sigmoid2"] = "sigmoid2",
        init_std: float = 0.0,
        min_std: float = 0.1,
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        self.space = space
        self.std_type = std_type
        self.min_std = min_std
        self.init_std = init_std

        out_features = get_out_features(space)

        self.mean_fc = layer_ctor(out_features)
        self.std_fc = layer_ctor(out_features)

        self._event_dims = len(space.shape)
        self.register_buffer("loc", 0.5 * (space.low + space.high))
        self.register_buffer("scale", 0.5 * (space.high - space.low))

        if init == "tf":
            self.apply(tf_init_)

    def forward(self, input: Tensor):
        mean: Tensor = self.mean_fc(input)
        mean = mean.reshape(-1, *self.space.shape)
        mean = F.tanh(mean)

        std: Tensor = self.std_fc(input)
        std = std.reshape(-1, *self.space.shape)
        std = process_std(std, self.std_type, self.init_std, self.min_std)

        dist = D.Normal(mean, std, self._event_dims)
        dist = D.TruncNormal(dist, -1.0, 1.0)
        dist = D.Affine(dist, self.loc, self.scale)

        return dist


class MSEProxy(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        self.space = space

        out_features = get_out_features(space)
        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)

        if init == "tf":
            self.apply(tf_init_)

    def forward(self, input: Tensor):
        value: Tensor = self.layer(input)
        value = value.reshape(-1, *self.space.shape)
        return D.MSEProxy(value, self._event_dims)


class Bernoulli(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Discrete,
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        assert space.n == 2 and space.dtype == torch.bool
        self.layer = layer_ctor(1)
        if init == "tf":
            self.apply(tf_init_)

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
        if getattr(self.layer, "bias") is not None:
            torch.nn.init.zeros_(self.layer.bias)
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
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        self.space = space
        self.vocab_size = int(space.shape[0])
        self.layer = layer_ctor(self.vocab_size)
        if init == "tf":
            self.apply(tf_init_)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.vocab_size)
        return D.OneHot(logits=logits)


class Discrete(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.TokenSeq,
        init: Literal["torch", "tf"] = "torch",
    ):
        super().__init__()
        self.space = space
        self.num_tokens, self.vocab_size = space.num_tokens, space.vocab_size
        self.layer = layer_ctor(space.num_tokens * space.vocab_size)
        if init == "tf":
            self.apply(tf_init_)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.num_tokens, self.vocab_size)
        return D.Discrete(logits=logits)


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
