from dataclasses import dataclass

from ..common import alpha
from ..common.utils import Optim


@dataclass
class Config:
    alpha: alpha.Config
    horizon: int
    update_epochs: int
    opt: Optim
    max_kl_div: float | None
    gamma: float
    gae_lambda: float
    update_batch: int
    adv_norm: bool
    clip_coeff: float
    clip_vloss: bool
    vf_coeff: float
    clip_grad: bool
    batch_size: int
