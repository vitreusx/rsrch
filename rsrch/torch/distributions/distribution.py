import torch
from torch import Tensor


class Distribution:
    batch_shape: tuple[int, ...]
    event_shape: tuple[int, ...]

    @property
    def mean(self) -> Tensor:
        return NotImplemented

    @property
    def variance(self) -> Tensor:
        return self.var

    @property
    def var(self) -> Tensor:
        return NotImplemented

    @property
    def std(self) -> Tensor:
        return self.var.sqrt()

    @property
    def mode(self) -> Tensor:
        return NotImplemented

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return NotImplemented

    def log_prob(self, value: Tensor) -> Tensor:
        return NotImplemented

    def entropy(self) -> Tensor:
        return NotImplemented

    def cdf(self, value: Tensor) -> Tensor:
        return NotImplemented
