from torch import Tensor
from torch.nn import *


class OneHot(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return functional.one_hot(x, self.num_classes)
