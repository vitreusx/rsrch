from rsrch.utils.config import *
from dataclasses import dataclass
from rsrch.rl.utils import make_env
from . import rssm


@dataclass
class Config:
    env: make_env.Config
    rssm: rssm.Config
    seq_len: int
    buf_cap: int
