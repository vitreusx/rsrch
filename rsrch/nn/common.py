import torch
from torch import nn, Tensor

from typing import Callable, Literal


ActType = Literal["relu", "elu", "tanh"]


def ActLayer(type: ActType) -> Callable[[], nn.Module]:
    return {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[type]


class LayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
    ):
        super().__init__()
        self._norm = nn.LayerNorm(
            normalized_shape=(num_features,),
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(self, input: Tensor):
        input = input.moveaxis(-3, -1)
        input = self._norm(input)
        input = input.moveaxis(-1, -3)
        return input


NormType = Literal["none", "batch", "layer"]


def NormLayer1d(type: NormType) -> Callable[[int], nn.Module]:
    return {
        "none": lambda _: nn.Identity(),
        "batch": nn.BatchNorm1d,
        "layer": nn.LayerNorm,
    }[type]


def NormLayer2d(type: NormType) -> Callable[[int], nn.Module]:
    return {
        "none": lambda _: nn.Identity(),
        "batch": nn.BatchNorm2d,
        "layer": LayerNorm2d,
    }[type]
