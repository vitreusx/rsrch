import torch
from torch import Tensor

from rsrch import spaces
from rsrch.types import Tensorlike

from .distribution import Distribution
from .one_hot_categorical import OneHotCategoricalST


class Particle(Distribution, Tensorlike):
    def __init__(self, space: spaces.torch.Box, size_logits: Tensor, logits: Tensor):
        event_shape = space.shape
        batch_shape = logits[: len(logits.shape) - len(event_shape) - 1]
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        offsets = size_logits.softmax(-1).cumsum(-1)
        offsets = torch.cat([torch.zeros_like(offsets[..., :1]), offsets], -1)
        pos = space.low + offsets * (space.high - space.low)
        self.pos = self.register("pos", pos)

        self.cat_rv = OneHotCategoricalST(logits=logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value)

    def entropy(self) -> Tensor:
        return super().entropy()

    def rsample(self, sample_shape: torch.Size = ()) -> Tensor:
        ind = self.cat_rv.rsample(sample_shape)
        return (ind * self.pos).sum(-1)
