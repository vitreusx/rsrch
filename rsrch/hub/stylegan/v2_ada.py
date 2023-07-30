import math
from typing import List, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.utils.eval_ctx import eval_ctx

_size_2_t = int | List[int]


def _fix_size(size: _size_2_t):
    if isinstance(size, int):
        size = [size, size]
    return torch.Size(size)


class ModConv2d(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        style_channels: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.style_channels = style_channels
        self.eps = eps

        self.conv = conv
        self.style_map = nn.Linear(self.style_channels, self.conv.in_channels)
        self.noise_scale = nn.Parameter(torch.randn([]))

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        mod_w = self.conv.weight * self.style_map(style).reshape(-1, 1, 1)
        demod_w = mod_w * (mod_w * mod_w + self.eps).rsqrt()
        input = self._conv(input, demod_w)
        noise = torch.randn(input.shape[-2:], dtype=input.dtype, device=input.device)
        return input + self.noise_scale * noise

    def _conv(self, input: Tensor, weight: Tensor) -> Tensor:
        _prev = self.conv.weight.data
        self.conv.weight.data = weight
        res = self.conv(input)
        self.conv.weight.data = _prev
        return res


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        upsample: str | None = None,
        conv_opts={},
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.upsample = upsample
        self.conv_opts = conv_opts

        if self.upsample is not None:
            self.up = nn.Upsample(scale_factor=2, mode=self.upsample)

        self.conv1 = ModConv2d(
            conv=nn.Conv2d(in_channels, out_channels, **conv_opts),
            style_channels=style_channels,
        )

        self.conv2 = ModConv2d(
            conv=nn.Conv2d(out_channels, out_channels, **conv_opts),
            style_channels=style_channels,
        )

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        if self.upsample is not None:
            input = self.up(input)
        input = self.conv1(input, style)
        input = self.conv2(input, style)
        return input


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        input_size: _size_2_t,
        channels: List[int],
        style_channels: int,
        conv_opts={},
    ):
        super().__init__()
        self.in_channels = channels[0]
        self.input_size = torch.Size([self.in_channels, *_fix_size(input_size)])
        self.channels = channels
        self.style_channels = style_channels
        self.conv_opts = conv_opts

        self.blocks = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.blocks.append(
                SynthesisBlock(
                    in_channels,
                    out_channels,
                    style_channels,
                    upsample=("bilinear" if len(self.blocks) > 0 else None),
                    conv_opts=conv_opts,
                )
            )

        self.in_channels = channels[0]
        with eval_ctx(self):
            inp = torch.empty(self.input_size).unsqueeze(0)
            style = torch.empty(self.style_channels).unsqueeze(0)
            out: Tensor = self(inp, style).squeeze(0)
        self.output_size = out.shape

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        for block in self.blocks:
            input = block(input, style)
        return input
