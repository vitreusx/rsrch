from typing import List, Protocol

import torch
from tensordict import TensorDict, tensorclass
from torch import Tensor

from rsrch.types.tensorlike import Tensorlike

from .distribution import Distribution
from .transforms import Transform


class TransformedDistribution(Distribution, Tensorlike):
    base: TensorDict
    event_shape: torch.Size

    def __init__(self, base: Distribution, *transforms: Transform):
        batch_shape = base.batch_shape
        Tensorlike.__init__(self, batch_shape)

        self.base: Distribution
        self.register("base", base)

        self.transforms = transforms

        x = base.sample()
        for t in transforms:
            x = t(x)
        self.event_shape = x.shape[len(base.batch_shape) :]

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        x = self.base.sample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        x = self.base.rsample(sample_shape)
        for t in self.transforms:
            x = t(x)
        return x

    def log_prob(self, value: Tensor) -> Tensor:
        logp = 0.0
        y = value
        event_dims = len(self.event_shape)

        for t in reversed(self.transforms):
            x = t.inv(value)
            t_logp = t.log_abs_det_jac(x, y).flatten(-event_dims).sum(-1)
            logp = logp + t_logp
            y = x

        base_logp = self.base.log_prob(y).flatten(-event_dims).sum(-1)
        logp = logp + base_logp
        return logp