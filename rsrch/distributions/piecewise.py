from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import Size, Tensor
from torch._C import Size

from rsrch import spaces
from rsrch.types import Tensorlike

from .categorical import Categorical
from .distribution import Distribution
from .utils import sum_rightmost


class Piecewise(Tensorlike, Distribution):
    def __init__(
        self,
        dist: Categorical,
        low: Number | Tensor,
        high: Number | Tensor,
        event_dims: int,
    ):
        Tensorlike.__init__(self, dist.shape)
        self.dist = self.register("dist", dist)
        self.k = self.dist.num_events
        self.low, self.high = low, high
        self.step = (high - low) / self.k
        self.event_dims = event_dims

    @property
    def mean(self):
        centers = torch.arange(self.k, dtype=float, device=self.device) + 0.5
        return (centers * self.dist.probs).mean(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        bucket = ((value - self.low) / self.step).floor().long()
        logp = self.dist.log_prob(bucket) - self.step.log()
        return sum_rightmost(logp, self.event_dims)

    def entropy(self) -> Tensor:
        ent = self.dist.entropy() + torch.log(self.high - self.low) - np.log(self.k)
        return sum_rightmost(ent, self.event_dims)

    def rsample(self, sample_size: tuple[int, ...] = tuple()):
        bucket = self.dist.sample(sample_size).float()
        bucket = bucket + torch.rand_like(bucket)
        return self.low + self.step * bucket

    def sample(self, sample_size: tuple[int, ...] = tuple()):
        with torch.no_grad():
            return self.rsample(sample_size)


class Piecewise3(Tensorlike, Distribution):
    def __init__(self, size_logits: Tensor, event_dims: int = 0):
        pivot = len(size_logits.shape) - 1 - event_dims
        batch_shape = size_logits.shape[:pivot]
        event_shape = size_logits.shape[pivot:-1]
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape
        self.num_q = size_logits.shape[-1]
        self.log_sizes = self.register("log_sizes", F.log_softmax(size_logits, -1))
        self.sizes = self.register("sizes", size_logits.softmax(-1))
        self.ends = self.register("ends", self.sizes.cumsum(-1))

    @property
    def mean(self):
        return (self.ends - 0.5 * self.sizes).sum(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        eps = 1e-6
        value = value.clamp(eps, 1 - eps)
        idxes = torch.searchsorted(self.ends, value.unsqueeze(-1))
        log_sizes = self.log_sizes.gather(-1, idxes).squeeze(-1)
        logp = -np.log(self.num_q) - log_sizes
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ent = np.log(self.num_q) + self.log_sizes.mean(-1)
        return sum_rightmost(ent, len(self.event_shape))

    def rsample(self, sample_size: tuple[int, ...] = ()):
        num_samples = int(np.prod(sample_size))
        shape = [num_samples, *self.sizes.shape]
        bucket: Tensor = torch.randint(
            0,
            self.num_q,
            [*shape[:-1], 1],
            device=self.device,
        )
        sizes = self.sizes[None].expand(shape)
        sizes = sizes.gather(-1, bucket).squeeze(-1)
        ends = self.ends[None].expand(shape)
        ends = ends.gather(-1, bucket).squeeze(-1)
        unif = torch.rand_like(ends)
        shape = [*sample_size, *self.batch_shape, *self.event_shape]
        return (ends - unif * sizes).reshape(shape)

    def sample(self, sample_size: tuple[int, ...] = ()):
        with torch.no_grad():
            return self.rsample(sample_size)


class Piecewise4(Distribution, Tensorlike):
    def __init__(self, space: spaces.torch.Box, size_logits: Tensor, logits: Tensor):
        event_shape = space.shape
        batch_shape = size_logits.shape[: len(size_logits.shape) - 1 - len(event_shape)]
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        loc, scale = space.low, space.high - space.low
        self.log_sizes = self.register(
            "log_sizes",
            size_logits.log_softmax(-1) + scale.log().unsqueeze(-1),
        )
        self.sizes = self.register("sizes", self.log_sizes.exp())
        self.ends = self.register("ends", loc.unsqueeze(-1) + self.sizes.cumsum(-1))

        self.ind_rv = self.register("ind_rv", Categorical(logits=logits))

    def log_prob(self, value: Tensor) -> Tensor:
        index = torch.searchsorted(self.ends, value.unsqueeze(-1))
        index = index.clamp_max(self.ends.shape[-1] - 1)
        log_pmf, log_sizes = self.ind_rv.log_probs, self.log_sizes
        index, log_pmf, log_sizes = torch.broadcast_tensors(index, log_pmf, log_sizes)
        index = index[..., :1]
        ind_logp = log_pmf.gather(-1, index).squeeze(-1)
        size_logp = -log_sizes.gather(-1, index).squeeze(-1)
        logp = ind_logp + size_logp
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        ind_ent = self.ind_rv.entropy()
        size_ent = (self.ind_rv.probs * self.log_sizes).sum(-1)
        ent = ind_ent + size_ent
        return sum_rightmost(ent, len(self.event_shape))

    def sample(self, sample_shape=()):
        index = self.ind_rv.sample(sample_shape)
        unif = torch.rand(index.shape, device=index.device)
        index = index.unsqueeze(-1)
        index, sizes, ends = torch.broadcast_tensors(index, self.sizes, self.ends)
        index = index[..., :1]
        sizes = sizes.gather(-1, index).squeeze(-1)
        ends = ends.gather(-1, index).squeeze(-1)
        return ends - sizes * unif

    def rsample(self, sample_shape=()):
        index = self.ind_rv.sample(sample_shape).unsqueeze(-1)
        unif = torch.rand(index.shape, device=index.device)
        log_pmf = self.ind_rv.log_probs
        index, unif, sizes, ends, log_pmf = torch.broadcast_tensors(
            index, unif, self.sizes, self.ends, log_pmf
        )
        index, unif = index[..., :1], unif[..., :1]
        sizes = sizes.gather(-1, index).squeeze(-1)
        ends = ends.gather(-1, index).squeeze(-1)
        log_pmf = log_pmf.gather(-1, index).squeeze(-1)
        return ends - sizes * unif + (log_pmf - log_pmf.detach())
