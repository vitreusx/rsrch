from typing import Any, Union
from dataclasses import dataclass


@dataclass
class Reset:
    env_idx: int
    obs: Any
    info: dict


@dataclass
class Step:
    env_idx: int
    act: Any
    next_obs: Any
    reward: float
    term: bool
    trunc: bool
    info: dict


Event = Union[Reset, Step]
