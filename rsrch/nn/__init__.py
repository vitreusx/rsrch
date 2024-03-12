import torch
from torch import Tensor
from torch.nn import *

from . import ensemble


class Reshape(Module):
    def __init__(self, shape: tuple[int, ...], start_dim=1, end_dim=-1):
        super().__init__()
        self.shape = shape
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        new_shape = x.shape[: self.start_dim] + self.shape + x.shape[self.end_dim :][1:]
        return x.reshape(new_shape)
