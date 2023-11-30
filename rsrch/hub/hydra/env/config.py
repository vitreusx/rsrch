from dataclasses import dataclass
from typing import Literal

from . import atari, gym


@dataclass
class Config:
    type: Literal["atari", "gym"]
    """Type of the environment. Either 'atari' or 'gym'."""
    atari: atari.Config
    gym: gym.Config
