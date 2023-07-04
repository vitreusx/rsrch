import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np
from torch.distributions import Categorical


class MaskedConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, mask: Tensor):
        super().__init__()
        self.mask: Tensor
        self.register_buffer("mask", mask.type_as(conv.weight))
        self.conv = conv

    def forward(self, x: Tensor) -> Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)


class InputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int,
        center_visible_channels: int,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._kernel_size = kernel_size
        self._center_visible_channels = center_visible_channels

        padding = kernel_size // 2
        self.conv = MaskedConv2d(
            conv=nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            mask=self._create_mask(),
        )

        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU()

    def _create_mask(self) -> Tensor:
        h, w = self._kernel_size, self._kernel_size
        mask = torch.ones((self._in_channels, h, w))
        mask[self._center_visible_channels :, h // 2, w // 2] = 0
        mask[:, h // 2, (w // 2 + 1) :] = 0
        mask[:, (h // 2 + 1) :] = 0
        return mask

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class HiddenLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size

        padding = kernel_size // 2
        self.conv = MaskedConv2d(
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            mask=self._create_mask(),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def _create_mask(self):
        h, w = self._kernel_size, self._kernel_size
        mask = torch.ones((h, w))
        mask[h // 2, (w // 2 + 1) :] = 0
        mask[(h // 2 + 1) :] = 0
        return mask

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class FinalLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, value_range: int):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._value_range = value_range

        self.conv = nn.Conv2d(
            in_channels,
            value_range * out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: Tensor):
        B, _, H, W = x.shape
        out: Tensor = self.conv(x)  # [B, N_out * N_val, H, W]
        out = out.reshape(B, self._out_channels, self._value_range, H, W)
        out = out.permute(0, 1, 3, 4, 2)  # [B, N_out, H, W, N_val]
        return out


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        value_range: int,
        kernel_size: int,
        num_layers: int,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._value_range = value_range
        self._kernel_size = kernel_size
        self._num_layers = num_layers

        def ChannelNet(channel_idx: int):
            center_visible_channels = channel_idx
            return nn.Sequential(
                InputLayer(
                    in_channels, hidden_dim, kernel_size, center_visible_channels
                ),
                *(
                    HiddenLayer(hidden_dim, hidden_dim, kernel_size)
                    for _ in range(num_layers - 1)
                ),
                FinalLayer(hidden_dim, 1, value_range)
            )

        self.nets = nn.ModuleList([ChannelNet(idx) for idx in range(in_channels)])

    def forward(self, x: Tensor):
        outs = [net(x) for net in self.nets]
        outs = torch.cat(outs, dim=1)  # [B, N_c, H, W, N_v]
        return outs

    def predict(self, x: Tensor, start: Tuple[int, int]):
        B, N, H, W = x.shape

        res = x.clone()

        self.eval()
        with torch.no_grad():
            for iy, ix, ic in np.ndindex((H, W, N)):
                if (iy, ix) < start:
                    continue
                logits: Tensor = self.nets[ic](res) # [B, 1, H, W, N_val]
                v = torch.argmax(logits[:, 0, iy, ix], dim=-1)  # [B,]
                res[:, ic, iy, ix] = v

        return res
