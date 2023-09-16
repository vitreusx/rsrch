from dataclasses import dataclass
from typing import Any, Sequence, Union
import numpy as np


@dataclass
class Reset:
    obs: Any
    info: dict


@dataclass
class Step:
    act: Any
    next_obs: Any
    reward: float
    term: bool
    trunc: bool
    info: dict


@dataclass
class VecReset:
    idxes: np.ndarray[int]
    obs: Any
    info: np.ndarray[dict]

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, i):
        return self.idxes[i], Reset(self.obs[i], self.info[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class VecStep:
    idxes: np.ndarray[int]
    act: Any
    next_obs: Any
    reward: np.ndarray[float]
    term: np.ndarray[bool]
    trunc: np.ndarray[bool]
    info: np.ndarray[dict]

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, i):
        step = Step(
            self.act[i],
            self.next_obs[i],
            self.reward[i],
            self.term[i],
            self.trunc[i],
            self.info[i],
        )
        return self.idxes[i], step

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class Async:
    pass


Event = Union[Reset, Step]
VecEvent = Union[VecReset, VecStep, Async]
