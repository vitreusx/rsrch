import jax
from jax import Array
from jax.nn import *

import math
from typing import Literal

from equinox import Module, field
from equinox.nn import *

from . import functional as F


def final():
    """Marks an attribute of Module as final. This disables any interaction
    with JAX transforms."""
    return field(static=True)


class Linear(Module):
    in_features: int = final()
    out_features: int = final()

    weight: Array
    bias: Array | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: jax.numpy.dtype | None = None,
        *,
        key: Array
    ):
        super().__init__()
        weight_key, bias_key = jax.random.split(key, 2)

        shape = (out_features, in_features)
        scale = 1.0 / math.sqrt(in_features)
        self.weight = jax.random.uniform(weight_key, shape, dtype, -scale, scale)

        if bias:
            shape = (out_features,)
            scale = 1.0 / math.sqrt(in_features)
            self.bias = jax.random.uniform(bias_key, shape, dtype, -scale, scale)
        else:
            self.bias = None

    @jax.jit
    def __call__(self, input: Array):
        return F.linear(input, self.weight, self.bias)


class Conv2d(Module):
    in_channels: int = final()
    out_channels: int = final()
    kernel_size: tuple[int, int] = final()
    stride: tuple[int, int] = final()
    padding: tuple[int, int] | Literal["same", "valid"] = final()
    dilation: tuple[int, int] = final()
    groups: int = final()

    weight: Array
    bias: Array | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | Literal["same", "valid"] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: jax.numpy.dtype | None = None,
        *,
        key: Array
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation
        self.groups = groups

        w_key, b_key = jax.random.split(key, 2)

        shape = (out_channels, in_channels // groups, *self.kernel_size)
        scale = int(math.prod(shape[1:]))
        scale = 1 / math.sqrt(scale)
        self.weight = jax.random.uniform(w_key, shape, dtype, -scale, scale)

        if bias:
            shape = (out_channels,)
            self.bias = jax.random.uniform(b_key, shape, dtype, -scale, scale)

    @jax.jit
    def __call__(self, input: Array):
        return F.conv_2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvTranspose2d(Module):
    in_channels: int = final()
    out_channels: int = final()
    kernel_size: tuple[int, int] = final()
    stride: tuple[int, int] = final()
    padding: tuple[int, int] | Literal["same", "valid"] = final()
    output_padding: tuple[int, int] = final()
    dilation: tuple[int, int] = final()

    weight: Array
    bias: Array | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | Literal["same", "valid"] = 0,
        output_padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        dtype: jax.numpy.dtype | None = None,
        *,
        key: Array
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        self.output_padding = output_padding
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        w_key, b_key = jax.random.split(key, 2)

        shape = (in_channels, out_channels, *self.kernel_size)
        scale = int(math.prod(shape[1:]))
        scale = 1 / math.sqrt(scale)
        self.weight = jax.random.uniform(w_key, shape, dtype, -scale, scale)

        if bias:
            shape = (out_channels,)
            self.bias = jax.random.uniform(b_key, shape, dtype, -scale, scale)

    @jax.jit
    def __call__(self, input: Array):
        return F.conv_transpose2d(
            input=input,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
        )


class ReLU(Module):
    @jax.jit
    def __call__(self, input: Array):
        return F.relu(input)


class ELU(Module):
    @jax.jit
    def __call__(self, input: Array):
        return F.elu(input)


class Sequential(Module):
    layers: tuple[Module]

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    @jax.jit
    def __call__(self, input: Array):
        for layer in self.layers:
            input = layer(input)
        return input


class Flatten(Module):
    start_dim: int = final()
    end_dim: int = final()

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @jax.jit
    def __call__(self, input: Array):
        start_dim = range(len(input.shape))[self.start_dim]
        end_dim = range(len(input.shape))[self.end_dim]
        return input.reshape(*input.shape[:start_dim], -1, *input.shape[end_dim + 1 :])
