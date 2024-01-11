from typing import Literal, Protocol

from rsrch.rl.gym import Env, VectorEnv

from . import atari, gym
from .config import Config


class Factory(Protocol):
    def env(
        self,
        mode: Literal["train", "val"] = "val",
        record: bool = False,
    ) -> Env:
        ...

    def vector_env(
        self,
        num_envs: int,
        mode: Literal["train", "val"] = "val",
    ) -> VectorEnv:
        ...


def make_factory(cfg: Config, device=None) -> Factory:
    if cfg.type == "atari":
        return atari.Factory(cfg.atari, device)
    elif cfg.type == "gym":
        return gym.Factory(cfg.gym, device)
    else:
        raise ValueError(f"Invalid env type: {cfg.type}")
