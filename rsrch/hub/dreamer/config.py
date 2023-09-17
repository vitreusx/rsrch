from dataclasses import dataclass

from rsrch.utils.config import *

from . import env, rssm


@dataclass
class Config:
    env: env.Config
    rssm: rssm.Config
    device: str
    val_every: int
    env_workers: int
