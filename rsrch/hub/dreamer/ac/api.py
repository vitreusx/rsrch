from torch import nn

import rsrch.distributions as D
from rsrch.rl import gym


class Actor(nn.Module):
    act_space: gym.spaces.TensorSpace

    def forward(self, state) -> D.Distribution:
        ...


class Critic(nn.Module):
    def forward(self, state) -> D.Distribution:
        ...
