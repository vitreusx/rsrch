from dataclasses import dataclass


@dataclass
class Config:
    env_steps: int
    epochs: int
    buffer_cap: int
    gamma: float
    tau: float
    lr: float
    min_ent: float | str
    batch_size: int
    prefill: int
    val_episodes: int
    max_ep_len: int
    env_name: str
