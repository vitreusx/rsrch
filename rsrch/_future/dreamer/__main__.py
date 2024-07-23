from pathlib import Path

from rsrch.utils import config

from .runner import Config, Runner


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")
    cfg = config.parse(cfg, Config)
    runner = Runner(cfg)
    runner.main()


if __name__ == "__main__":
    main()
