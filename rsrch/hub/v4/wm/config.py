from dataclasses import dataclass
from typing import Literal
from . import rssm


@dataclass
class Config:
    type: Literal["rssm", "exact"]
    rssm: rssm.Config
