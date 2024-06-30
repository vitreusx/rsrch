from functools import partial
from typing import Literal

from jax.nn import *
from optax.losses import *

import rsrch.jax.numpy as jnp
from rsrch import jax
from rsrch.jax import Array, amp, lax

softmax = amp.autocast_to_fp32(softmax)
log_softmax = amp.autocast_to_fp32(log_softmax)
softplus = amp.autocast_to_fp32(softplus)
cosine_similarity = amp.autocast_to_fp32(cosine_similarity)
softmax_cross_entropy = amp.autocast_to_fp32(softmax_cross_entropy)
kl_divergence = amp.autocast_to_fp32(kl_divergence)


@amp.autocast_to_fp16
def linear(input: Array, weight: Array, bias: Array | None = None):
    output = weight @ input
    if bias is not None:
        output = output + bias
    return output


@partial(jax.jit, static_argnames=["stride", "padding", "dilation", "groups"])
@amp.autocast_to_fp16
def conv_2d(
    input: Array,
    weight: Array,
    bias: Array | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | Literal["same", "valid"] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        if isinstance(padding, int):
            padding = (padding, padding)

        pad_h, pad_w = padding
        padding = [(pad_h, pad_h), (pad_w, pad_w)]

    output = lax.conv_general_dilated(
        lhs=input,
        rhs=weight,
        window_strides=stride,
        padding=padding,
        lhs_dilation=dilation,
        feature_group_count=groups,
    )

    if bias is not None:
        bias = bias.reshape(*bias.shape, 1, 1)
        output = output + bias

    return output


@partial(jax.jit, static_argnames=["stride", "padding", "output_padding", "dilation"])
@amp.autocast_to_fp16
def conv_transpose2d(
    input: Array,
    weight: Array,
    bias: Array | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    kernel_size = weight.shape[-2:]

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        if isinstance(padding, int):
            padding = (padding, padding)

        pad_h, pad_w = padding
        pad_h = dilation[0] * (kernel_size[0] - 1) - pad_h
        pad_w = dilation[1] * (kernel_size[1] - 1) - pad_w

        padding = [(pad_h, pad_h), (pad_w, pad_w)]

    output = lax.conv_transpose(
        lhs=input,
        rhs=input,
        strides=stride,
        padding=padding,
        rhs_dilation=dilation,
    )

    if bias is not None:
        bias = bias.reshape(*bias.shape, 1, 1)
        output = output + bias

    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)

    out_pad_h, out_pad_w = output_padding
    pad_width = [(0, out_pad_h), (0, out_pad_w)]
    output = jnp.pad(output, pad_width)

    return output


@partial(jax.jit, static_argnames=["normalized_shape", "eps"])
@amp.autocast_to_fp32
def layer_norm(
    input: Array,
    normalized_shape: tuple[int, ...],
    weight: Array | None = None,
    bias: Array | None = None,
    eps: float = 1e-5,
):
    input_ = input.reshape(-1, *normalized_shape)
    mean, var = input_.mean(0), input_.var(0)
    output = (input - mean) / (jnp.sqrt(var) + eps)
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias
    return output
