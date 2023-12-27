from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rsrch.utils import _config

from . import dreamerx, env, ppo, td3


@dataclass
class Config:
    env: env.Config
    type: Literal["ppo", "td3", "dreamerx"]


def main():
    cfg = _config.cli(
        Config,
        config_yml=Path(__file__).parent / "config.yml",
        presets_yml=Path(__file__).parent / "presets.yml",
        # args=["-h"],
        args=["-p", "cont", "--env.type", "gym", "--env.gym.env_id", "Ant-v4"],
        # args=["-p", "HalfCheetah-v4"],
        # args=["-p", "Alien-v5"],
    )

    if cfg.type == "ppo":
        ppo.main()
    elif cfg.type == "td3":
        td3.main()
    elif cfg.type == "dreamerx":
        dreamerx.main()


if __name__ == "__main__":
    main()
