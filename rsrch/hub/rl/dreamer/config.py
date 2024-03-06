from dataclasses import dataclass

from rsrch.utils.config import *

from .. import env


@dataclass
class Config:
    env: env.Config
