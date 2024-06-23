import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        bias: bool = True,
        *,
        input_layer: int | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if input_layer is None:
            input_layer = in_channels
        self.input_layer = input_layer

        self.mask: Tensor
        self.register_buffer("mask", self._make_mask())

    def _make_mask(self):
        I = torch.meshgrid([torch.arange(d) for d in self.weight.shape])
        _, ic, ix = I
        cx = self.kernel_size[0] // 2
        mask = ic < self.input_layer
        mask = (ix < cx) | ((ix == cx) & mask)
        return mask.float().type_as(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(
            input,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CausalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        bias: bool = True,
        *,
        input_layer: int | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if input_layer is None:
            input_layer = in_channels
        self.input_layer = input_layer

        self.mask: Tensor
        self.register_buffer("mask", self._make_mask())

    def _make_mask(self):
        I = torch.meshgrid([torch.arange(d) for d in self.weight.shape])
        _, ic, iy, ix = I
        cy, cx = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        mask = ic < self.input_layer
        mask = (ix < cx) | ((ix == cx) & mask)
        mask = (iy < cy) | ((iy == cy) & mask)
        return mask.float().type_as(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
