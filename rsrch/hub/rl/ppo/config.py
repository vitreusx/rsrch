from dataclasses import dataclass
from typing import Optional

from rsrch import rl
from rsrch.utils.config import *
from . import trainer


@dataclass
class Net:
    share_encoder: bool
    custom_init: bool


@dataclass
class Config:
    seed: int
    env: rl.sdk.Config
    val_episodes: int
    train_envs: int
    steps_per_epoch: int
    device: str
    compute_dtype: str
    total_steps: int
    log_every: int
    save_ckpt_every: int
    val_every: int
    net: Net
    trainer: trainer.Config
