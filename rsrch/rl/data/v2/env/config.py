from dataclasses import dataclass
from typing import Literal

from . import atari as atari_
from . import gym as gym_

__all__ = ["Config"]


@dataclass
class Config:
    type: Literal["atari", "gym"]
    atari: atari_.Config | None = None
    gym: gym_.Config | None = None

    @property
    def id(self):
        return getattr(self, self.type).env_id
