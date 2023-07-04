import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        act=nn.ReLU,
        use_bn=True,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._act = act
        self._use_bn = use_bn

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act1 = act()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act2 = act()
        self.down = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.down(x), x


class Middle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        act=nn.ReLU,
        use_bn=True,
        mode="bilinear",
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._act = act
        self._use_bn = use_bn
        self._mode = mode

        self.up = nn.Upsample(scale_factor=2, mode=mode)
        self.conv1 = nn.Conv2d(
            2 * in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act1 = act()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act2 = act()

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        act=nn.ReLU,
        use_bn=True,
        mode="bilinear",
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._act = act
        self._use_bn = use_bn
        self._mode = mode

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
        )
        self.conv1 = nn.Conv2d(
            2 * in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act1 = act()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act2 = act()

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = torch.cat([z, self.up(x)], dim=1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: List[int]):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._hidden = hidden

        down_channels = [in_channels, *hidden[:-1]]
        self.down_layers = []
        for c_in, c_out in zip(down_channels[:-1], down_channels[1:]):
            down_layer = Down(c_in, c_out)
            self.down_layers.append(down_layer)

        self.mid_block = Middle(hidden[-2], hidden[-1])

        up_channels = [*reversed(hidden)]
        self.up_layers = []
        for c_in, c_out in zip(up_channels[:-1], up_channels[1:]):
            up_layer = Up(c_in, c_out)
            self.up_layers.append(up_layer)

        self.final_conv = nn.Conv2d(
            hidden[0], out_channels, 3, padding="same", bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        zs = []
        for layer in self.down_layers:
            x, z = layer(x)
            zs.append(z)

        x = self.mid_block(x)
        for layer, z in zip(self.up_layers, reversed(zs)):
            x = layer(x, z)

        x = self.final_conv(x)
        return x
