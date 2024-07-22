import math
from typing import Callable, Literal

from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces


class Normal(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Space,
        std_type: Literal["const", "exp", "softplus"] = "softplus",
    ):
        super().__init__()
        if not self.space.dtype.is_floating_point:
            raise ValueError()
        self.space = space
        self.norm_std = std_type

        out_features = math.prod(space.shape)
        if std_type != "const":
            out_features *= 2
        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)

    def forward(self, input: Tensor):
        output: Tensor = self.layer(input)
        if self.norm_std != "const":
            output = output.reshape(-1, 2, *self.space.shape)
            mean, std = output.unbind(1)
        else:
            mean = output.reshape(-1, *self.space.shape)
            std = 1.0
        return D.Normal(mean, std, self._event_dims)


class Beta(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
    ):
        super().__init__()
        self.space = space

        out_features = 2 * math.prod(space.shape)
        self.layer = layer_ctor(out_features)

        self._event_dims = len(space.shape)
        self.register_buffer("loc", space.low)
        self.register_buffer("scale", space.high - space.low)

    def forward(self, input: Tensor):
        output: Tensor = self.layer(input)
        output = output.reshape(-1, 2, *self.space.shape)
        alpha, beta = output.unbind(1)
        alpha, beta = 1.0 + alpha.exp(), 1.0 + beta.exp()
        dist = D.Beta(alpha, beta, self._event_dims)
        dist = D.Affine(dist, self.loc, self.scale)
        return dist


class MSEProxy(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Box,
    ):
        super().__init__()
        self.space = space

        out_features = math.prod(space.shape)
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
    ):
        super().__init__()
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

        self.layer = layer_ctor(space.n)
        self._event_dims = len(space.shape)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.space.n)
        return D.Categorical(logits=logits, event_dims=self._event_dims)


class OneHot(nn.Module):
    def __init__(
        self,
        layer_ctor: Callable[[int], nn.Module],
        space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.space = space
        self.layer = layer_ctor(space.n)

    def forward(self, input: Tensor):
        logits: Tensor = self.layer(input)
        logits = logits.reshape(-1, self.space.n)
        return D.OneHot(logits=logits)


DistType = Literal["auto", "normal", "beta", "mse", "bern", "cat", "one_hot"]


def make(
    layer_ctor: Callable[[int], nn.Module],
    space: spaces.torch.Space,
    type: DistType = "auto",
    **kwargs,
):
    if type == "auto":
        if isinstance(space, spaces.torch.Discrete):
            type = "bern" if space.n <= 2 else "cat"
        elif isinstance(space, spaces.torch.Image):
            type = "mse"
        elif isinstance(space, spaces.torch.Box):
            type = "beta" if space.bounded.all() else "normal"

    cls = {
        "normal": Normal,
        "beta": Beta,
        "mse": MSEProxy,
        "bern": Bernoulli,
        "cat": Categorical,
        "one_hot": OneHot,
    }[type]

    return cls(layer_ctor, space, **kwargs.get(type, {}))
