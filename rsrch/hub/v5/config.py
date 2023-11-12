from dataclasses import dataclass

from rsrch.exp import profiler
from rsrch.utils.config import *

from . import env, hybrid


@dataclass
class Config:
    device: str
    algo: Literal["dreamer", "cem"]
    wm: Literal["hybrid"]
    env_workers: int
    val_episodes: int
    val_every: int
    train_envs: int
    buffer_cap: int
    log_every: int
    total_steps: int
    prefill: int
    env_steps: int
    opt_steps: int

    env: env.Config
    hybrid: hybrid.Config
    profiler: profiler.Config
