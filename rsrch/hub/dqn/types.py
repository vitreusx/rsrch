from typing import Protocol
from torch import Tensor


class QNetwork(Protocol):
    num_actions: int

    def __call__(self, obs: Tensor) -> Tensor:
        ...
