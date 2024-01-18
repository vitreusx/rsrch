from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rsrch.utils import config

from . import dreamerx, env, ppo, redq, sac, td3


@dataclass
class Config:
    env: env.Config
    type: str


def main():
    cfg = config.cli(
        Config,
        config_yml=Path(__file__).parent / "config.yml",
        presets_yml=Path(__file__).parent / "presets.yml",
        # args=["-h"],
        args=["-p", "cont", "--env.type", "gym", "--env.gym.env_id", "Ant-v4"],
        # args=["-p", "HalfCheetah-v4"],
        # args=["-p", "Alien-v5"],
    )

    globals()[cfg.type].main()


if __name__ == "__main__":
    main()
