from typing import Generic, Protocol, TypeAlias, TypeVar

from torch import FloatTensor

import rsrch.distributions as D

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Agent(Protocol, Generic[ObsType, ActType]):
    def reset(self):
        ...

    def act(self, obs: ObsType) -> ActType:
        ...


class Policy(Protocol, Generic[ObsType, ActType]):
    def __call__(self, obs: ObsType) -> D.Distribution:
        ...


Actor: TypeAlias = Policy


class VFunc(Protocol, Generic[ObsType]):
    def __call__(self, obs: ObsType) -> float | FloatTensor:
        ...


Critic: TypeAlias = VFunc


class QFunc(Protocol, Generic[ObsType, ActType]):
    def __call__(self, obs: ObsType, act: ActType) -> float | FloatTensor:
        ...
