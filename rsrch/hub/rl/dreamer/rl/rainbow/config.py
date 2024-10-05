from dataclasses import dataclass


@dataclass
class Dist:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float


@dataclass
class Noisy:
    enabled: bool
    sigma0: float
    factorized: bool


@dataclass
class Config:
    encoder: dict
    hidden_dim: int
    gamma: float
    double_dqn: bool
    dist: Dist
    noisy: Noisy
    opt: dict
    grad_clip: float | None
    dueling: bool
    polyak: dict
