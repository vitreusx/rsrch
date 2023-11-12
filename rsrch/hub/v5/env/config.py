from dataclasses import dataclass
from typing import Literal
from . import atari, dmc, gym


@dataclass
class Config:
    type: Literal["atari", "gym", "dmc"]
    atari: atari.Config
    gym: gym.Config
    dmc: dmc.Config
