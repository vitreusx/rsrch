import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ruamel import yaml

from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler import Profiler
from rsrch.exp.wandb import Experiment
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron

from . import config, env, hybrid
from .common.utils import flat
from .config import Config


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg
        device = torch.device(cfg.device)
        env_f = env.make_factory(cfg.env, device)

        exp = Experiment(project="v5", config=asdict(self.cfg))
        self.board = exp.board
        env_step = 0
        self.board.add_step("env_step", lambda: env_step, default=True)
        self.should_log = cron.Every(lambda: env_step, cfg.log_every)
        should_end = cron.Once(lambda: env_step >= cfg.total_steps)
        should_val = cron.Every(lambda: env_step, cfg.val_every)
        pbar = ProgressBar(desc="V5", total=cfg.total_steps)

        prof = Profiler(
            cfg=cfg.profiler,
            device=device,
            step_fn=lambda: env_step,
            trace_path=exp.dir / "trace.json",
        )

        sampler = data.UniformSampler()

        if cfg.wm == "hybrid":
            wm = hybrid.impl.WorldModel(
                cfg.hybrid.impl, env_f.obs_space, env_f.act_space
            ).to(device)

            wm_trainer = hybrid.wm.Trainer(cfg.hybrid.wm, wm, ctx=self)

            buf = env_f.chunk_buffer(cfg.buffer_cap, cfg.hybrid.wm.seq_len, sampler)

            if cfg.algo == "dreamer":
                actor = hybrid.impl.Actor(cfg.hybrid.impl, env_f.act_space).to(device)
                agent_fn = lambda: hybrid.wm.VecAgent(wm, actor)
                critic_ctor = lambda: hybrid.impl.Critic(cfg.hybrid.impl).to(device)
                algo_trainer = hybrid.dreamer.Trainer(
                    cfg.hybrid.dreamer,
                    wm,
                    env_f.act_space,
                    actor,
                    critic_ctor,
                    ctx=self,
                )
            elif cfg.algo == "cem":
                planner = hybrid.cem.CEMPlanner(cfg.hybrid.cem, wm, env_f.act_space)
                agent_fn = lambda: hybrid.wm.VecAgent(wm, planner)

        val_envs = env_f.vector_env(cfg.env_workers, mode="val")
        val_agent = env_f.VecAgent(agent_fn(), memoryless=False)

        train_envs = env_f.vector_env(cfg.train_envs, mode="train")
        train_agent = gym.vector.agents.EpsAgent(
            opt=env_f.VecAgent(agent_fn(), memoryless=False),
            rand=gym.vector.agents.RandomAgent(train_envs),
            eps=1.0,
            per_batch=True,
        )
        train_iter = iter(rollout.steps(train_envs, train_agent))
        ep_ids = [None for _ in range(train_envs.num_envs)]

        def update_env_step(n: int = 1):
            nonlocal env_step
            env_step += n
            pbar.step(n)
            train_agent.eps = 1.0 - env_step / cfg.total_steps

        while True:
            if should_val:
                val_rets = []
                for _, ep in rollout.episodes(
                    val_envs, val_agent, max_episodes=cfg.val_episodes
                ):
                    val_rets.append(sum(ep.reward))

                self.board.add_scalar("val/returns", np.mean(val_rets))

            if should_end:
                break

            for _ in range(cfg.env_steps):
                env_idx, step = next(train_iter)
                ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
                update_env_step()
                prof.update()

            for _ in range(cfg.opt_steps):
                if len(buf) < cfg.prefill:
                    continue

                idxes = sampler.sample(cfg.hybrid.wm.batch_size)
                batch = env_f.fetch_chunk_batch(buf, idxes)
                states = wm_trainer.opt_step(batch)

                if cfg.algo == "dreamer":
                    init_s = flat(states[1:-1])
                    n = cfg.hybrid.dreamer.batch_size
                    init_s = init_s[torch.randint(len(init_s), size=(n,))]
                    algo_trainer.opt_step(init_s)

                prof.update()


def main():
    dicts = []
    presets = ["CartPole-v1"]

    with open(Path(__file__).parent / "config.yml", "r") as f:
        def_cfg = yaml.safe_load(f)
        dicts.append(def_cfg)

    with open(Path(__file__).parent / "presets.yml", "r") as f:
        pre_cfg = yaml.safe_load(f) or {}
        for preset in presets:
            dicts.append(pre_cfg.get(preset, {}))

    cfg_dict = config.to_dict(dicts)
    cfg = config.to_class(cfg_dict, Config)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
