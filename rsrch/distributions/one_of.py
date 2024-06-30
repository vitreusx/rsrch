import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .categorical import Categorical
from .distribution import Distribution
from .one_hot import OneHot


class OneOf(Tensorlike, Distribution):
    def __init__(
        self,
        choices: Tensor,
        *,
        logits: Tensor | None = None,
        probs: Tensor | None = None,
    ):
        Tensorlike.__init__(self, logits.shape[:-1])
        self.event_shape = choices.shape[1:]
        self.onehot = self.register("onehot", OneHot(probs=probs, logits=logits))
        self.choices = self.register("choices", choices, batched=False)

    def sample(self, sample_shape=()):
        onehot = self.onehot.sample(sample_shape)
        return torch.tensordot(onehot, self.choices, 1)

    def rsample(self, sample_shape=()):
        onehot = self.onehot.rsample(sample_shape)
        return torch.tensordot(onehot, self.choices, 1)

    def entropy(self) -> Tensor:
        return self.onehot.entropy()

    def log_prob(self, value: Tensor):
        indices = torch.cdist(value[None], self.choices[None]).argmax(-1)[0]
        return self.onehot.index_dist.log_prob(indices)
