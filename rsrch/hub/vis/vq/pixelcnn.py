from typing import Literal
from torch import nn, Tensor
import torch
import numpy as np
import torch.nn.functional as F

_size_2_t = int | tuple[int, int]


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros"] = "zeros",
        device=None,
        dtype=None,
        first_layer=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        n_out, n_in, ker_h, ker_w = self.weight.shape
        i_out, i_in, i_y, i_x = np.mgrid[:n_out, :n_in, :ker_h, :ker_w]
        c_y, c_x = ker_h // 2, ker_w // 2
        mask = i_y < c_y
        mask |= (i_y == c_y) & (i_x < c_x)
        mask |= (i_y == c_y) & (i_x == c_x) & (i_in < i_out)
        if not first_layer:
            mask |= (i_y == c_y) & (i_x == c_x) & (i_in == i_out)

        mask = torch.as_tensor(mask).type_as(self.weight)
        self.mask: Tensor
        self.register_buffer("mask", mask)

    def forward(self, input: Tensor):
        return F.conv2d(
            input,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class PixelCNNBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.res = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(in_features, in_features // 2, 1, padding=0),
            nn.ReLU(),
            MaskedConv2d(in_features // 2, in_features // 2, 3, padding=1),
            nn.ReLU(),
            MaskedConv2d(in_features // 2, in_features, 1, 1, padding=0),
        )

    def forward(self, input: Tensor):
        return input + self.res(input)


class PixelCNN(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim=128,
        num_layers=15,
    ):
        super().__init__(
            MaskedConv2d(in_features, hidden_dim, 7, padding=3, first_layer=True),
            nn.Sequential(*(PixelCNNBlock(hidden_dim) for _ in range(num_layers))),
            nn.ReLU(),
            MaskedConv2d(hidden_dim, out_features, 1),
        )
