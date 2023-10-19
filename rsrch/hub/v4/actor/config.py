from dataclasses import dataclass
from typing import Literal
from . import sac, ppo, mpc


@dataclass
class Config:
    type: Literal["sac"]
    sac: sac.Config
    # type: Literal["sac", "ppo", "mpc"]
    # ppo: ppo.Config
    # mpc: mpc.Config
