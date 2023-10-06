from dataclasses import dataclass
from typing import Literal

from . import sac


@dataclass
class Config:
    type: Literal["sac"]
    sac: sac.Config
