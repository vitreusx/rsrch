from dataclasses import dataclass
from typing import Optional
from rsrch.utils.config import *


@dataclass
class Env:
    @dataclass
    class Atari:
        screen_size: int
        frame_skip: int
        term_on_life_loss: bool
        grayscale: bool
        noop_max: int
        frame_stack: Optional[int]
        fire_reset: bool
        episodic_life: bool

    name: str
    type: str
    atari: Atari
    reward: str | tuple[int, int]
    time_limit: Optional[int]


@dataclass
class Config:
    env: Env
    env_workers: int
    val_episodes: int
    val_every: int
    train_envs: int
    steps_per_epoch: int
    device: str
    gamma: float
    gae_lambda: float
    update_epochs: int
    update_batch: int
    adv_norm: bool
    clip_coeff: float
    clip_vloss: bool
    lr: float
    opt_eps: float
    clip_grad: Optional[float]
    total_steps: int
    log_every: int
    ent_coeff: float
    vf_coeff: float
    decorrelate: bool
    share_encoder: bool
    custom_init: bool
    seed: int
