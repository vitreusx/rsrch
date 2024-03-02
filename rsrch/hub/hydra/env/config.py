from dataclasses import dataclass
from typing import Literal

from . import atari as Atari
from . import dmc as DMC
from . import gym as Gym

__all__ = ["Config"]


@dataclass
class Config:
    type: Literal["atari", "gym", "dmc"]
    atari: Atari.Config | None = None
    gym: Gym.Config | None = None
    dmc: DMC.Config | None = None
