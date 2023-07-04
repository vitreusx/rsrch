import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Union
import numpy as np


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
        self.register_buffer("mask", mask)
        self.conv = conv
        assert self.conv.kernel_size == self.mask.shape

    def forward(self, x: Tensor) -> Tensor:
        self.conv.weight.data *= self.mask
        return self.conv(x)


# class RowLSTM(nn.Module):
#     def __init__(self, in_features: int, hidden_dim: int, kernel_size: int):
#         super().__init__()
#         self.in_features = in_features
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size

#         self.input_conv = MaskedConv(
#             conv=nn.Conv2d(in_features, 4 * hidden_dim, kernel_size),
#             mask=MaskFactory().causal_2d((kernel_size, kernel_size)),
#         )
#         self.state_conv = nn.Conv1d(hidden_dim, 4 * hidden_dim, kernel_size)

#     def forward(
#         self, x: Tensor, h_c: Tuple[Tensor, Tensor]
#     ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
#         B, C, H, W = x.shape
#         Kx = self.input_conv(x)

#         h, c = h_c
#         hs = torch.empty(B, self.hidden_dim, H, W)
#         for row in range(H):
#             gates = Kx[:, :, row] + self.state_conv(h)
#             i, f, g, o = torch.split(gates, self.hidden_dim)
#             c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
#             h = torch.sigmoid(o) * c
#             hs[:, :, row] = h

#         return hs, (h, c)


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        kernel_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        layers = []
        for layer_idx in range(self.num_layers):
            in_features_ = in_features if layer_idx == 0 else hidden_dim
            conv = nn.Conv2d(in_features_, hidden_dim, kernel_size, 1, kernel_size//2, bias=False)
            mask = MaskFactory().causal_init(conv.kernel_size)
            conv = MaskedConv(conv, mask)
            layers.extend([conv, nn.BatchNorm2d(hidden_dim), nn.ReLU()])
        layers.append(nn.Conv2d(hidden_dim, out_features, 1, bias=True))

        self._main = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._main(x)
