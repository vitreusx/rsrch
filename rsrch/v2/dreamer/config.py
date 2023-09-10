from rsrch.utils.config import *
from dataclasses import dataclass
from rsrch.rl.utils import make_env
from . import rssm


@dataclass
class Config:
    env: make_env.Config
    rssm: rssm.Config
    seq_len: int
    batch_size: int
    horizon: int
    buf_cap: int
    val_episodes: int
    env_workers: int
    device: str
    val_every: int
    total_steps: int
    train_envs: int
