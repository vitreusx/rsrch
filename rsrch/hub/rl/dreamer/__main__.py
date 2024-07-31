import warnings
from pathlib import Path

from rsrch.utils import config

from .runner import Config, Runner


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")

    presets = config.load(Path(__file__).parent / "presets.yml")
    config.add_preset_(cfg, presets, "default")

    cfg = config.parse(cfg, Config)
    runner = Runner(cfg)
    runner.main()


if __name__ == "__main__":
    main()
