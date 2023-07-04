import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal


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


class Encoder2d(nn.Module):
    def __init__(self, x_shape: torch.Size, z_shape: torch.Size):
        super().__init__()
