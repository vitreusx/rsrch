import warnings
from pathlib import Path

from rsrch.utils import config

from .runner import Config, Runner


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")

    presets = config.load(Path(__file__).parent / "presets.yml")
    preset_names = ["atari"]
    for name in preset_names:
        config.add_preset_(cfg, presets, name)

    cfg = config.parse(cfg, Config)
    runner = Runner(cfg)
    runner.main()


if __name__ == "__main__":
    main()
