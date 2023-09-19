import numpy as np
import torch
from .distribution import Distribution
from rsrch.types import Tensorlike
from rsrch.types.tensorlike import TensorTuple
import torch


class Ensemble(Distribution, Tensorlike):
    def __init__(self, rvs: list[Distribution]):
        Tensorlike.__init__(self, rvs[0].batch_shape)
        self.event_shape = rvs[0].event_shape
        self._n = len(rvs)
        self.rvs = self.register("rvs", TensorTuple(rvs))

    def rsample(self, sample_size: torch.Size = torch.Size()):
        num_samples = int(np.prod(sample_size))
        if num_samples == 1:
            idx = np.random.randint(self._n)
            return self.rvs.as_tuple()[idx].rsample(sample_size)

        idxes = torch.randint(0, self._n, [num_samples])
        counts = torch.zeros(self._n, dtype=torch.long)
        counts.index_add_(0, idxes, torch.tensor(1).expand_as(idxes))

        samples = []
        for idx, c in enumerate(counts):
            if c > 0:
                samples.append(self.rvs.as_tuple()[idx].rsample([c]))

        samples = torch.cat(samples, 0)
        samples = samples[torch.randperm(num_samples)]
        samples = samples.reshape([*sample_size, *samples.shape[1:]])
        return samples

    def sample(self, sample_size: torch.Size = torch.Size()):
        num_samples = int(np.prod(sample_size))
        if num_samples == 1:
            idx = np.random.randint(self._n)
            return self.rvs.as_tuple()[idx].sample(sample_size)

        idxes = torch.randint(0, self._n, [num_samples])
        counts = torch.zeros(self._n, dtype=torch.long)
        counts.index_add_(0, idxes, torch.tensor(1).expand_as(idxes))

        samples = []
        for idx, c in enumerate(counts):
            if c > 0:
                samples.append(self.rvs.as_tuple()[idx].sample([c]))

        samples = torch.cat(samples, 0)
        samples = samples[torch.randperm(num_samples)]
        samples = samples.reshape([*sample_size, *samples.shape[1:]])
        return samples
