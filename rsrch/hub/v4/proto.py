import rsrch.distributions as D
from torch import nn


class WorldModel:
    def step(self, s, a) -> D.Distribution:
        ...

    def term(self, s) -> D.Distribution:
        ...

    def reward(self, next_s) -> D.Distribution:
        ...


class Actor(nn.Module):
    def __call__(self, s) -> D.Distribution:
        ...


class Critic(nn.Module):
    def __call__(self, s) -> D.Distribution:
        ...
