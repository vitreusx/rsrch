import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Union


class MaskFactory:
    def full(self, kernel_size: Tuple[int, ...]) -> Tensor:
        return torch.ones(kernel_size)

    def causal_init(self, kernel_size: Tuple[int, int]) -> Tensor:
        h, w = kernel_size
        mask = torch.ones((h, w))
        mask[h // 2, (w // 2) :] = 0
        mask[(h // 2 + 1) :] = 0
        return mask

    def causal_next(self, kernel_size: Tuple[int, int]) -> Tensor:
        h, w = kernel_size
        mask = torch.ones((h, w))
        mask[h // 2, (w // 2 + 1) :] = 0
        mask[(h // 2 + 1) :] = 0
        return mask


class MaskedConv(nn.Module):
    def __init__(self, conv: Union[nn.Conv1d, nn.Conv2d], mask: Tensor):
        super().__init__()
        self.mask: Tensor
        self.register_buffer("mask", mask)
        self.conv = conv
        assert self.conv.kernel_size == self.mask.shape

    def forward(self, x: Tensor) -> Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        value_range: int,
        kernel_size: int,
        num_layers: int,
    ):
        super().__init__()
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._out_channels = out_channels
        self._value_range = value_range
        self._kernel_size = kernel_size
        self._num_layers = num_layers

        def MaskedConvBlock(in_channels: int, out_channels: int):
            padding = kernel_size // 2
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, bias=False
            )
            mask = MaskFactory().causal_init(conv.kernel_size)
            conv = MaskedConv(conv, mask)
            return nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self._final_nc = out_channels * value_range
        channels = [
            in_channels,
            *([hidden_dim] * (num_layers - 1)),
        ]

        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.append(MaskedConvBlock(in_channels, out_channels))

        final_conv = nn.Conv2d(hidden_dim, self._final_nc, 1)
        layers.append(final_conv)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Categorical:
        B, _, H, W = x.shape
        out: Tensor = self.layers(x)
        out = out.reshape(B, self._out_channels, self._value_range, H, W)
        out = out.permute(0, 1, 3, 4, 2)  # Move value_range channel to end
        return Categorical(logits=out)
