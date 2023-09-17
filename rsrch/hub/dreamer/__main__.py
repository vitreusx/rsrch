from pathlib import Path

import torch

from rsrch.exp import Experiment
from rsrch.utils import cron

from . import config, env, rssm, wm


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    loader = env.Loader(cfg.env)

    dreamer = rssm.Dreamer(loader.val_env(), cfg.rssm)
    dreamer = dreamer.to(device)

    val_envs = loader.val_envs(cfg.env_workers)

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
