from dataclasses import dataclass

from rsrch.rl import sdk

from . import alpha


@dataclass
class Random:
    seed: int
    deterministic: bool


@dataclass
class Polyak:
    every: int
    tau: float


@dataclass
class Value:
    opt: dict
    polyak: Polyak
    sched: dict


@dataclass
class Actor:
    opt: dict
    opt_ratio: int


@dataclass
class Val:
    envs: int
    sched: dict
    episodes: int


@dataclass
class Config:
    random: Random
    device: str
    compute_dtype: str
    env: sdk.Config
    total_steps: int
    num_envs: int
    batch_size: int
    log_sched: dict
    warmup: int
    buf_cap: int
    gamma: float
    value: Value
    actor: Actor
    alpha: alpha.Config
    hidden_dim: int
    val: Val
