import warnings
from pathlib import Path

from rsrch.utils import config

from .runner import Config, Runner


def main():
    cfg = config.cli(
        config_yml=Path(__file__).parent / "config.yml",
        presets_yml=Path(__file__).parent / "presets.yml",
        def_presets=["default"],
    )
    cfg = config.cast(cfg, Config)

    runner = Runner(cfg)
    runner.main()


if __name__ == "__main__":
    main()
