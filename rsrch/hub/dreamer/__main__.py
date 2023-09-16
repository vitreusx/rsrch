from . import config
from pathlib import Path


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )


if __name__ == "__main__":
    main()
