import torch
import torch.nn as nn
from typing import Any, Protocol


class Transform(Protocol):
    def __call__(self, x: Any) -> Any:
        ...


class Compose(nn.Module, Transform):
    def __init__(self, *transforms: Transform):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
