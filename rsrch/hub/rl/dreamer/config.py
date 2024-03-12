from rsrch.utils.config import *

from .. import env


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Config:
    device: str
    random: Random
    env: env.Config
    buf_cap: int
    slice_len: int
