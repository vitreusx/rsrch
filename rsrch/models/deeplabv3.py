"""DeepLabV3 from [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)."""

import inspect
from copy import deepcopy
from typing import Literal, Type, TypeVar

import torch
import torchvision.transforms.functional as tv_F
from torch import Tensor, nn

from rsrch.models import resnet
from rsrch.models.types import ActLayer, NormLayer

T = TypeVar("T")


def conv2d_set_dilation(conv: nn.Conv2d, dilation: int):
    """Set conv layer's dilation, also setting other parameters up so that
    the output size is the same as the input size."""

    conv.stride = (1, 1)
    conv.dilation = (dilation, dilation)
    # See [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) for conv output size arithmetic.
    pad_x = dilation * (conv.kernel_size[0] - 1) // 2
    pad_y = dilation * (conv.kernel_size[1] - 1) // 2
    conv.padding = (pad_x, pad_y)


def walk(module: nn.Module):
    """Walk through a module tree."""

    yield module
    for child in module.children():
        yield from walk(child)


def convert_resnet_(
    model: resnet.Resnet,
    dilations: list[int | list[int]],
):
    """Replace strides in the final couple layers of ResNet model with dilations.

    The length of `dilations` indicates how many of the final layers to take. If `dilations[idx]` is a list, different dilation rates are applied to consecutive blocks in the layer. This is the multi-grid method in Subsection 3.2.1.
    """

    num_layers = len(dilations)
    for layer, dilation in zip(model.layers[-num_layers:], dilations):
        layer: nn.Sequential
        if isinstance(dilation, int):
            dilation = [dilation] * len(layer)
        for block, rate in zip(layer, dilation):
            for node in walk(block):
                if isinstance(node, nn.Conv2d):
                    conv2d_set_dilation(node, rate)


class ConvModule(nn.Sequential):
    """A conv->[norm->]act module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | Literal["same"] = "same",
        dilation: int = 1,
        groups: int = 1,
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
    ):
        if norm_layer is None:
            bias = True
            norm = None
        else:
            bias = False
            norm = norm_layer(out_channels)

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if norm is None:
            super().__init__(conv, act_layer())
        else:
            super().__init__(conv, norm, act_layer())


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: list[int],
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
        interpolation: tv_F.InterpolationMode = tv_F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.interpolation = interpolation

        # Reading the paper, it's not at all clear whether to use simple conv,
        # or conv->[norm->]act sequence. Following torchvision, I use the latter.
        conv1x1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        atrous = []
        for rate in dilations:
            atrous.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                act_layer=act_layer,
                norm_layer=norm_layer,
            ),
        )

        self.branches = nn.ModuleList([conv1x1, *atrous, pool])

        self.proj = ConvModule(
            out_channels * len(self.branches),
            out_channels,
            kernel_size=1,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, input: Tensor):
        outputs = []
        for branch in self.branches:
            output = branch(input)
            output = tv_F.resize(output, input.shape[-2:], self.interpolation)
            outputs.append(output)
        return self.proj(torch.cat(outputs, 1))
