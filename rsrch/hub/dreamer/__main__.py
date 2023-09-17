from pathlib import Path

import numpy as np
import torch

from rsrch.exp import Experiment
from rsrch.rl.data import rollout
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

    dreamer = rssm.Dreamer(cfg.rssm, loader.obs_space, loader.act_space)
    dreamer = dreamer.to(device)

    wm_opt = cfg.opt.wm(dreamer.wm.parameters())
    ac_opt = cfg.opt.ac([*dreamer.actor.parameters(), *dreamer.critic.parameters()])

    num_val_envs = min(cfg.env_workers, cfg.val_episodes)
    val_envs = loader.val_envs(num_val_envs)
    val_agent = wm.VecEnvAgent(dreamer.wm, dreamer.actor, val_envs)
    val_agent = loader.VecAgent()(val_agent)

    exp_envs = loader.exp_envs(cfg.exp_envs)
    exp_agent = wm.VecEnvAgent(dreamer.wm, dreamer.actor, exp_envs)
    exp_agent = loader.VecAgent()(exp_agent)
    exp_iter = rollout.events(exp_envs, exp_agent)

    exp = Experiment(project="dreamer")
    board = exp.board
    env_step = 0
    board.add_step("env_step", lambda: env_step, default=True)

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_stop = cron.Once(lambda: env_step >= cfg.total_steps)

    def val_epoch():
        val_iter = rollout.episodes(val_envs, val_agent, max_episodes=cfg.val_episodes)
        val_returns = [sum(ep.reward) for _, ep in val_iter]
        board.add_scalar("val/returns", np.mean(val_returns))

    while True:
        if should_val:
            val_epoch()

        if should_stop:
            break


if __name__ == "__main__":
    main()
