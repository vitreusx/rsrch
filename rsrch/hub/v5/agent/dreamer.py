from dataclasses import dataclass

from ..common.utils import Optim, Polyak


@dataclass
class Config:
    @dataclass
    class Alpha:
        autotune: bool
        ent_scale: float | None
        value: float | None
        opt: Optim

    @dataclass
    class Coefs:
        critic: float
        actor_pg: float
        actor_v: float

    batch_size: int
    horizon: int
    gamma: float
    gae_lambda: float
    adv_norm: bool
    alpha: Alpha
    polyak: Polyak
    coefs: Coefs
    opt: Optim
