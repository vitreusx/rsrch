from pathlib import Path

from rsrch.exp import Experiment
from rsrch.utils import cron

from . import config


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    exp = Experiment(project="dreamer")
    board = exp.board
    env_step = 0
    board.add_step("env_step", lambda: env_step, default=True)

    should_val = cron.Every(lambda: env_step, cfg.val_every)

    def val_epoch():
        ...

    while True:
        if should_val:
            val_epoch()


if __name__ == "__main__":
    main()
