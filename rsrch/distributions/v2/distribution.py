from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Distribution(Protocol):
    event_shape: torch.Size
    batch_shape: torch.Size

    mean: Tensor
    variance: Tensor
    std: Tensor
    mode: Tensor

    def sample(self, sample_shape: torch.Size) -> Tensor:
        ...

    def rsample(self, sample_shape: torch.Size) -> Tensor:
        ...

    def log_prob(self, value) -> Tensor:
        ...

    def entropy(self) -> Tensor:
        ...
