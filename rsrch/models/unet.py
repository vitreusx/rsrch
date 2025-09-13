import itertools

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from torch import Tensor, nn

from rsrch.models.types import ActLayer, NormLayer


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
    ):
        super().__init__()

        if norm_layer is None:
            bias = True
            norm1 = norm2 = None
        else:
            bias = False
            norm1 = norm_layer(out_channels)
            norm2 = norm_layer(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = norm1
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = norm2
        self.act2 = act_layer()
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, input: Tensor):
        x = self.conv1(input)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.conv2(self.act1(x))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.act2(x)
        out, skip = self.pool(x), x
        return out, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
    ):
        super().__init__()

        if norm_layer is None:
            bias = True
            norm1 = norm2 = None
        else:
            bias = False
            norm1 = norm_layer(out_channels)
            norm2 = norm_layer(out_channels)

        # Original UNet has "up-conv 2x2", but such with such a kernel, one
        # cannot preserve image size. Therefore we use 3x3 kernel
        self.up_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels, out_channels, 3, 1, 1, bias=bias
        )
        self.norm1 = norm1
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = norm2
        self.act2 = act_layer()

    def forward(self, input: Tensor, skip: Tensor):
        input = self.up_conv(F.interpolate(input, scale_factor=2))
        skip = tv_F.resize(skip, input.shape[-2:])
        input = torch.cat((skip, input), 1)

        x = self.conv1(input)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.conv2(self.act1(x))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.act2(x)

        return x


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
    ):
        super().__init__()

        if norm_layer is None:
            bias = True
            norm1 = norm2 = None
        else:
            bias = False
            norm1 = norm_layer(out_channels)
            norm2 = norm_layer(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm1 = norm1
        self.act1 = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.norm2 = norm2
        self.act2 = act_layer()

    def forward(self, input: Tensor):
        x = self.conv1(input)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.conv2(self.act1(x))
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.act2(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_channels: list[int],
        act_layer: type[ActLayer] = nn.ReLU,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
    ):
        super().__init__()

        down_channels = [in_channels, *block_channels[:-1]]
        self.down_blocks = nn.ModuleList()
        for in_, out_ in itertools.pairwise(down_channels):
            self.down_blocks.append(
                DownBlock(
                    in_channels=in_,
                    out_channels=out_,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        self.mid_block = MidBlock(
            in_channels=down_channels[-1],
            out_channels=block_channels[-1],
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        up_channels = block_channels[::-1]
        self.up_blocks = nn.ModuleList()
        for in_, out_ in itertools.pairwise(up_channels):
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_,
                    out_channels=out_,
                    skip_channels=out_,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        self.out_conv = nn.Conv2d(block_channels[0], out_channels, 1, 1, 0)

    def forward(self, input: Tensor):
        skip_features = []
        for block in self.down_blocks:
            input, skip = block(input)
            skip_features.append(skip)

        input = self.mid_block(input)

        for block, skip in zip(self.up_blocks, skip_features[::-1], strict=False):
            input = block(input, skip)

        return self.out_conv(input)
