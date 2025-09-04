from functools import partial
from typing import Any, Literal, overload

import equinox as eqx
import jax
from jax import Array

from rsrch.jax.utils import key_seq


class NormLayer(eqx.Module):
    """API for a norm layer."""

    def __init__(self, input_size: int):
        pass

    @overload
    def __call__(self, input: Array) -> Array:
        pass

    @overload
    def __call__(self, input: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        pass


class ActLayer(eqx.Module):
    """API for an act layer."""

    def __init__(self):
        pass

    @overload
    def __call__(self, input: Array) -> Array:
        pass

    @overload
    def __call__(self, input: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        pass


ReLU = lambda: jax.nn.relu
BatchNorm2d = partial(
    eqx.nn.BatchNorm,
    axis_name="batch",
    momentum=0.9,  # Pytorch uses 0.9 rather than 0.99
    mode="batch",  # Pytorch doesn't use EMA
)


def call_stateful(
    layer: eqx.Module, input: Array, state: eqx.nn.State
) -> tuple[Any, eqx.nn.State]:
    """Call a layer, passing the state along if it's stateful.

    Depending on statefulness of `layer`, either
    `output, state = layer(input), state` or `output, state = layer(input, state)`
    must be called. This function provides a uniform interface for both cases."""

    if isinstance(layer, eqx.nn.StatefulLayer) and layer.is_stateful():
        input, state = layer(input, state)
    else:
        input = layer(input)
    return input, state


class BasicBlock(eqx.nn.StatefulLayer):
    conv1: eqx.nn.Conv
    bn1: eqx.Module | None
    act: eqx.Module
    conv2: eqx.nn.Conv
    bn2: eqx.Module | None
    downsample: eqx.Module | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: type[NormLayer] | None = BatchNorm2d,
        act_layer: type[ActLayer] = ReLU,
        *,
        key: Array,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        # Residual path
        if norm_layer is None:
            bn1 = bn2 = None
            bias = True
        else:
            bn1 = norm_layer(out_channels)
            bn2 = norm_layer(out_channels)
            bias = False

        keys = key_seq(key)

        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, 3, stride, 1, use_bias=bias, key=next(keys)
        )
        self.bn1 = bn1
        self.act = act_layer()
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels, 3, 1, 1, use_bias=bias, key=next(keys)
        )
        self.bn2 = bn2

        # Skip path
        if in_channels != out_channels or stride > 1:
            conv = eqx.nn.Conv2d(
                in_channels, out_channels, 1, stride, use_bias=bias, key=next(keys)
            )
            if norm_layer is None:
                self.downsample = conv
            else:
                norm = norm_layer(out_channels)
                self.downsample = eqx.nn.Sequential([conv, norm])
        else:
            self.downsample = None

    def __call__(self, input: Array, state: eqx.nn.State, *, key=None):
        res = self.conv1(input)
        if self.bn1 is not None:
            res, state = call_stateful(self.bn1, res, state)
        res, state = call_stateful(self.act, res, state)
        res = self.conv2(res)
        if self.bn2 is not None:
            res, state = call_stateful(self.bn2, res, state)

        if self.downsample is not None:
            input, state = call_stateful(self.downsample, input, state)

        return input + res, state


class Bottleneck(eqx.nn.StatefulLayer):
    conv1: eqx.nn.Conv2d
    bn1: eqx.Module | None
    act1: eqx.Module
    conv2: eqx.nn.Conv2d
    bn2: eqx.Module | None
    act2: eqx.Module
    bn3: eqx.Module | None
    downsample: eqx.Module | None

    def __init__(
        self,
        in_channels: int,
        bottleneck: int,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: type[NormLayer] | None = BatchNorm2d,
        act_layer: type[ActLayer] = ReLU,
        *,
        key: Array,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        # Residual path
        if norm_layer is None:
            bn1 = bn2 = bn3 = None
            bias = True
        else:
            bn1 = norm_layer(bottleneck)
            bn2 = norm_layer(bottleneck)
            bn3 = norm_layer(out_channels)
            bias = False

        keys = key_seq(key)

        # Original ResNet adds stride in conv1, but torchvision adds it on
        # conv2. See
        #   https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        # for the rationale

        self.conv1 = eqx.nn.Conv2d(
            in_channels, bottleneck, 1, use_bias=bias, key=next(keys)
        )
        self.bn1 = bn1
        self.act1 = act_layer()
        self.conv2 = eqx.nn.Conv2d(
            bottleneck, bottleneck, 3, stride, 1, use_bias=bias, key=next(keys)
        )
        self.bn2 = bn2
        self.act2 = act_layer()
        self.conv3 = eqx.nn.Conv2d(
            bottleneck, out_channels, 1, use_bias=bias, key=next(keys)
        )
        self.bn3 = bn3

        # Skip path
        if in_channels != out_channels or stride > 1:
            conv = eqx.nn.Conv2d(
                in_channels, out_channels, 1, stride, use_bias=bias, key=next(keys)
            )
            if norm_layer is None:
                self.downsample = conv
            else:
                norm = norm_layer(out_channels)
                self.downsample = eqx.nn.Sequential([conv, norm])
        else:
            self.downsample = None

    def __call__(self, input: Array, state: eqx.nn.State, *, key=None):
        res = self.conv1(input)
        if self.bn1 is not None:
            res, state = call_stateful(self.bn1, res, state)
        res, state = call_stateful(self.act1, res, state)
        res = self.conv2(res)
        if self.bn2 is not None:
            res, state = call_stateful(self.bn2, res, state)
        res, state = call_stateful(self.act2, res, state)
        res = self.conv3(res)
        if self.bn3 is not None:
            res, state = call_stateful(self.bn3, res, state)

        if self.downsample is not None:
            input, state = call_stateful(self.downsample, input, state)

        return input + res, state


class Resnet(eqx.nn.StatefulLayer):
    num_blocks: list[int] = eqx.field(static=True)
    num_channels: list[int] = eqx.field(static=True)
    block_type: Literal["basic", "bottleneck"] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    num_classes: int | None = eqx.field(static=True)

    conv1: eqx.nn.Conv2d
    bn1: eqx.Module | None
    relu: eqx.Module
    maxpool: eqx.nn.MaxPool2d
    layers: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d | None
    fc: eqx.nn.Linear | None

    def __init__(
        self,
        num_blocks: list[int],
        num_channels: list[int],
        block_type: Literal["basic", "bottleneck"] = "basic",
        in_channels: int = 3,
        num_classes: int | None = 1000,
        norm_layer: type[eqx.Module] | None = BatchNorm2d,
        act_layer: type[eqx.Module] = ReLU,
        *,
        key: Array,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.block_type = block_type
        self.in_channels = in_channels
        self.num_classes = num_classes

        assert len(num_blocks) == len(num_channels)

        if norm_layer is None:
            bn1 = None
            bias = True
        else:
            bn1 = norm_layer(64)
            bias = False

        keys = key_seq(key)

        self.conv1 = eqx.nn.Conv2d(
            in_channels, 64, 7, 2, 3, use_bias=bias, key=next(keys)
        )
        self.bn1 = bn1
        self.relu = act_layer()
        self.maxpool = eqx.nn.MaxPool2d(3, 2, 1)

        if block_type == "bottleneck":
            bottleneck_sizes = [nc // 4 for nc in num_channels]
        else:
            bottleneck_sizes = [None] * len(num_blocks)

        self.layers = eqx.nn.Sequential(
            [
                self._make_layer(
                    in_channels=64 if idx == 0 else num_channels[idx - 1],
                    bottleneck=bottleneck_sizes[idx],
                    out_channels=num_channels[idx],
                    num_blocks=num_blocks[idx],
                    downsample=(idx > 0),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=next(keys),
                )
                for idx in range(len(num_blocks))
            ]
        )

        if num_classes is not None:
            self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = eqx.nn.Linear(num_channels[3], num_classes, key=next(keys))
        else:
            self.avgpool = self.fc = None

    def _make_layer(
        self,
        in_channels: int,
        bottleneck: int | None,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
        norm_layer: type[eqx.Module],
        act_layer: type[eqx.Module],
        *,
        key: Array,
    ):
        layers = []
        for block_idx in range(num_blocks):
            stride = 2 if downsample and block_idx == 0 else 1
            if bottleneck is None:
                layer = BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=key,
                )
            else:
                layer = Bottleneck(
                    in_channels=in_channels,
                    bottleneck=bottleneck,
                    out_channels=out_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=key,
                )
            layers.append(layer)
            in_channels = out_channels

        return eqx.nn.Sequential(layers)

    def __call__(self, input: Array, state: eqx.nn.State):
        input = self.conv1(input)
        if self.bn1 is not None:
            input, state = call_stateful(self.bn1, input, state)
        input, state = call_stateful(self.relu, input, state)
        input = self.maxpool(input)

        input, state = self.layers(input, state)

        if self.num_classes is not None:
            input = self.avgpool(input)
            output = self.fc(input.flatten())
        else:
            output = input

        return output, state


def _resnet_factory(func):
    def wrapped(
        in_channels: int = 3,
        num_classes: int = 1000,
        norm_layer: type[NormLayer] | None = BatchNorm2d,
        act_layer: type[ActLayer] = ReLU,
        *,
        key: Array,
    ) -> Resnet:
        return func(
            in_channels=in_channels,
            num_classes=num_classes,
            norm_layer=norm_layer,
            act_layer=act_layer,
            key=key,
        )

    return wrapped


@_resnet_factory
def resnet18(**kwargs):
    return Resnet(
        num_blocks=[2, 2, 2, 2],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
        **kwargs,
    )


@_resnet_factory
def resnet34(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
        **kwargs,
    )


@_resnet_factory
def resnet50(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


@_resnet_factory
def resnet101(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 23, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


@_resnet_factory
def resnet152(**kwargs):
    return Resnet(
        num_blocks=[3, 8, 36, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


class MNIST(eqx.Module):
    layers: list

    def __init__(self, in_channels: int = 1, num_classes: int = 10, *, key: Array):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(in_channels, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jax.numpy.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, num_classes, key=key4),
        ]

    def __call__(self, x: Array, state: eqx.nn.State) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x, state


def mnist(
    in_channels: int = 1,
    num_classes: int = 10,
    *,
    key: Array,
):
    return MNIST(in_channels, num_classes, key=key)
