from typing import List

import numpy as np
import torch

from rsrch.types.tensorlike import Tensorlike

from .categorical import Categorical
from .distribution import Distribution
from .one_hot_categorical import OneHotCategoricalST


class Mixture(Distribution, Tensorlike):
    def __init__(self, index: Categorical, values: List[Distribution]):
        batch_shape = index.batch_shape
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = values[0].event_shape

        self.index: Categorical
        self.register("index", index)

        self._num_values = len(values)
        for val_idx in range(len(values)):
            self.register(f"value_{val_idx}", values[val_idx])

        self._init_from_fields()

    def _init_from_fields(self):
        self._ohst = OneHotCategoricalST(
            probs=self.index._probs,
            logits=self.index._logits,
        )
        self.values = [
            getattr(self, f"value_{val_idx}") for val_idx in range(self._num_values)
        ]

    def _new(self, shape: torch.Size, fields: dict):
        new = super()._new(shape, fields)
        new._init_from_fields()
        return new

    @property
    def mean(self):
        mx = [val.mean for val in self.values]
        px = [self.index.probs[..., val_idx] for val_idx in range(len(self.values))]
        return torch.stack([p * m for p, m in zip(px, mx)]).sum(0)

    @property
    def variance(self):
        mx = [val.mean for val in self.values]
        vx = [val.variance for val in self.values]
        px = [self.index.probs[..., val_idx] for val_idx in range(len(self.values))]
        t1 = torch.stack([p * (v + m**2) for p, v, m in zip(px, vx, mx)]).sum(0)
        t2 = torch.stack([p * m for p, m in zip(px, mx)]).sum(0).square()
        return t1 - t2

    def sample(self, sample_size: torch.Size = torch.Size()):
        idx = self.index.sample(sample_size).ravel()
        idx = idx.reshape(len(idx), *self.event_shape)

        samples = []
        for val in self.values:
            val_sample = val.sample(sample_size).flatten(0, -len(val.event_shape))
            samples.append(val_sample)
        samples = torch.stack(samples)

        samples = samples.gather(0, idx)

        target_shape = [*sample_size, *self.batch_shape, *self.event_shape]
        samples = samples.reshape(target_shape)
        return samples

    def rsample(self, sample_size: torch.Size = torch.Size()):
        sample_size = torch.Size(sample_size)
        S = sample_size.numel()
        B = self.batch_shape.numel()
        E = self.event_shape.numel()
        N = len(self.values)

        onehot = self._ohst.rsample(sample_size)
        onehot = onehot.reshape(S, B, N)

        samples = []
        for val in self.values:
            val_sample = val.sample(sample_size)
            samples.append(val_sample)
        samples = torch.stack(samples)
        samples = samples.reshape(N, S, B, E)

        samples = torch.einsum("sbn,nsbe->sbe", onehot, samples)

        target_shape = [*sample_size, *self.batch_shape, *self.event_shape]
        samples = samples.reshape(target_shape)
        return samples

    def log_prob(self, value):
        logpx = [val.log_prob(value) for val in self.values]
        logpx = torch.stack(logpx, -1)
        return torch.logsumexp(logpx * self.index.logits, -1)
