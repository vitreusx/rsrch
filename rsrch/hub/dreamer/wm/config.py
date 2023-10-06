from dataclasses import dataclass
from typing import Literal

from . import rssm


@dataclass
class Config:
    type: Literal["rssm", "test"]
    rssm: rssm.Config
