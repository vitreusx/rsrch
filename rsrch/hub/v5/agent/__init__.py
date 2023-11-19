from dataclasses import dataclass
from typing import Literal

from . import cem, dreamer, ppo


@dataclass
class Config:
    type: Literal["cem", "dreamer", "ppo"]
    cem: cem.Config
    dreamer: dreamer.Config
    ppo: ppo.Config
