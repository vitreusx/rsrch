import torch
from torch import Tensor


class Distribution:
    event_shape: torch.Size
    batch_shape: torch.Size

    @property
    def mean(self) -> Tensor:
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        return self.var

    @property
    def var(self) -> Tensor:
        raise NotImplementedError()

    @property
    def std(self) -> Tensor:
        return self.var.sqrt()

    @property
    def mode(self) -> Tensor:
        raise NotImplementedError()

    def sample(self, sample_shape: torch.Size = ()) -> Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size = ()) -> Tensor:
        raise NotImplementedError()

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError()

    def entropy(self) -> Tensor:
        raise NotImplementedError()

    def cdf(self, value: Tensor) -> Tensor:
        raise NotImplementedError()
