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


@dataclass
class VecStep:
    idxes: np.ndarray[int]
    act: Any
    next_obs: Any
    reward: np.ndarray[float]
    term: np.ndarray[bool]
    trunc: np.ndarray[bool]
    info: np.ndarray[dict]


@dataclass
class Async:
    pass


Event = Union[Reset, Step]
VecEvent = Union[VecReset, VecStep, Async]
