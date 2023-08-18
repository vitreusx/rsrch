import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym


class Random:
    def __init__(self, act_space: gym.Space):
        self._act_space = act_space
        if isinstance(self._act_space, gym.spaces.TensorBox):
            self._dist = D.Uniform(self._act_space.low, self._act_space.high)
        elif isinstance(self._act_space, gym.spaces.TensorDiscrete):
            logits = torch.zeros(self._act_space.shape, device=self._act_space.device)
            self._dist = D.OneHotCategoricalST(logits=logits)

    def actor(self, x: Tensor) -> D.Distribution:
        batch_size = x.shape[0]
        dist = self._dist.expand(batch_size, *self._dist.shape)
        return dist
