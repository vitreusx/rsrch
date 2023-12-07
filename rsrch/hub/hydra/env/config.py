from dataclasses import dataclass
from typing import Literal

from . import atari as atari_
from . import gym as gym_


@dataclass
class Config:
    type: Literal["atari", "gym"]
    """Type of the environment. Either 'atari' or 'gym'."""
    atari: atari_.Config | None = None
    gym: gym_.Config | None = None