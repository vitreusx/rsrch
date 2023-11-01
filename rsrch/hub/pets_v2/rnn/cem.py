from dataclasses import dataclass


@dataclass
class Config:
    niters: int
    pop: int
    elites: int
    horizon: int
