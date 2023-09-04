from dataclasses import dataclass
from typing import Optional
from rsrch.rl.utils.make_env import EnvConfig


@dataclass
class Config:
    env: EnvConfig
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
