from abc import ABC, abstractmethod
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from torch import FloatTensor

import rsrch.distributions as D

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Agent(Protocol, Generic[ObsType, ActType]):
    def reset(self):
        pass

    def observe(self, obs: ObsType):
        pass

    @abstractmethod
    def policy(self) -> ActType:
        ...

    def step(self, act: ActType):
        pass


class Policy(Protocol, Generic[ObsType, ActType]):
    def __call__(self, obs: ObsType) -> D.Distribution:
        ...


Actor: TypeAlias = Policy


class Critic(Protocol, Generic[ObsType]):
    def __call__(self, obs: ObsType) -> float | FloatTensor:
        ...


class QValue(Protocol, Generic[ObsType, ActType]):
    def __call__(self, obs: ObsType, act: ActType) -> float | FloatTensor:
        ...
