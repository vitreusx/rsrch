from dataclasses import dataclass

from rsrch.exp import profiler
from rsrch.utils.config import *

from . import agent, deter, env, hybrid


@dataclass
class Config:
    @dataclass
    class WM:
        type: Literal["hybrid", "deter"]
        hybrid: hybrid.Config
        deter: deter.Config

    device: str
    env_workers: int
    val_episodes: int
    val_every: int | None
    train_envs: int
    buffer_cap: int
    log_every: int
    total_steps: int
    prefill: int
    env_steps: int
    opt_steps: int

    env: env.Config
    wm: WM
    agent: agent.Config
    profiler: profiler.Config
