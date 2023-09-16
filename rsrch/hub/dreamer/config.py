from rsrch.utils.config import *
from dataclasses import dataclass
from . import env


@dataclass
class Config:
    env: env.Config
