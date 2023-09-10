import numpy as np
import torch
from . import config, rssm, wm
from rsrch.rl.utils import make_env
from rsrch.rl import gym, data
from rsrch.rl.data import rollout
from pathlib import Path
from rsrch.exp.board.wandb import Wandb
from rsrch.exp.vcs import WandbVCS
from rsrch.utils import cron
from rsrch.exp.pbar import ProgressBar


T = gym.wrappers.transforms


def main():
    cfg = config.from_args(
        cls=config.Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)

    env_f = make_env.EnvFactory(cfg.env, record_stats=True, to_tensor=True)

    def make_env_(env_fn):
        def _env_fn():
            env = env_fn()
            if cfg.env.stack:
                env = gym.wrappers.Apply(env, T.Concat(env.observation_space))
            return env

        return _env_fn

    make_train_env = make_env_(env_f.train_env)
    make_val_env = make_env_(env_f.val_env)

    env_spec = gym.EnvSpec(make_val_env())

    nets = rssm.AllNets(env_spec, cfg.rssm)
    nets = nets.to(device)

    if cfg.train_envs > 1:
        train_env = gym.vector.AsyncVectorEnv2(
            [make_train_env] * cfg.train_envs,
            num_workers=cfg.env_workers,
        )
    else:
        train_env = gym.vector.SyncVectorEnv([make_train_env])

    val_env = gym.vector.AsyncVectorEnv2(
        env_fns=[make_val_env] * cfg.val_episodes,
        num_workers=cfg.env_workers,
    )

    train_agent = wm.VecEnvAgent(nets.wm, nets.actor, train_env, device)
    val_agent = wm.VecEnvAgent(nets.wm, nets.actor, val_env, device)

    buffer = data.ChunkBuffer(
        nsteps=cfg.seq_len,
        capacity=cfg.buf_cap,
        stack_in=cfg.env.stack,
        persist=data.TensorStore(cfg.buf_cap),
    )

    env_iter = iter(rollout.steps(train_env, train_agent))
    ep_id = [None for _ in range(train_env.num_envs)]

    env_step = 0
    should_val = cron.Every(lambda: env_step, cfg.val_every)

    pbar = ProgressBar(total=cfg.total_steps)
    board = Wandb(project="dreamer")
    vcs = WandbVCS()
    vcs.save()

    def val_epoch():
        val_returns = []
        val_eps = rollout.episodes(val_env, val_agent, max_episodes=cfg.val_episodes)
        for _, ep in val_eps:
            val_returns.append(sum(ep.reward))
        board.add_scalar("val/returns", np.mean(val_returns), step=env_step)

    def train_step():
        nonlocal env_step
        env_idx, step = next(env_iter)
        ep_id[env_idx], _ = buffer.push(ep_id[env_idx], step)
        env_step += 1
        pbar.update()

    while True:
        if should_val:
            val_epoch()
        if env_step > cfg.total_steps:
            break
        train_step()


if __name__ == "__main__":
    main()
