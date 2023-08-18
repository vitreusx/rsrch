from dataclasses import dataclass
from typing import Optional

from rsrch.utils.sched import schedule


@dataclass
class Env:
    name: str
    screen_size: int
    frame_stack: int
    frame_skip: int
    reward_clip: tuple[float, float]
    term_on_loss_of_life: bool
    max_frames_per_episode: int


@dataclass
class Sched:
    num_frames: int
    val_every: int
    opt_batch: int
    env_batch: int
    replay_ratio: float


@dataclass
class Buffer:
    prefill: int
    capacity: int


@dataclass
class Encoder:
    type: str
    variant: Optional[str]


@dataclass
class Optimizer:
    name: str
    lr: float
    eps: float


@dataclass
class NoisyNets:
    enabled: bool
    sigma0: float
    factorized: bool


@dataclass
class MultiStep:
    enabled: bool
    n: int


@dataclass
class Dist:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float


@dataclass
class Pr:
    enabled: bool
    alpha: float
    beta: float | schedule


@dataclass
class Config:
    env: Env
    sched: Sched
    buffer: Buffer
    encoder: Encoder
    optimizer: Optimizer
    noisy_nets: NoisyNets
    multi_step: MultiStep
    dist: Dist
    pr: Pr
    gamma: float
