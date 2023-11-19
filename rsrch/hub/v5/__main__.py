import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ruamel import yaml

from rsrch.exp import comet
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler import Profiler
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.utils import cron

from . import config, deter, env, hybrid
from .common.utils import flat
from .config import Config


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        cfg = self.cfg
        device = torch.device(cfg.device)
        env_f = env.make_factory(cfg.env, device)

        self.exp = comet.Experiment(project="v5")
        env_step = 0
        self.exp.register_step("env_step", lambda: env_step, default=True)
        self.should_log = cron.Every(lambda: env_step, cfg.log_every)
        should_end = cron.Once(lambda: env_step >= cfg.total_steps)
        should_val = cron.Every(lambda: env_step, cfg.val_every)
        pbar = ProgressBar(desc="V5", total=cfg.total_steps, leave=True)

        prof = Profiler(
            cfg=cfg.profiler,
            device=device,
            step_fn=lambda: env_step,
            trace_path=self.exp.dir / "trace.json",
        )

        sampler = data.UniformSampler()

        if cfg.wm.type == "hybrid":
            wm = hybrid.impl.WorldModel(
                cfg=cfg.wm.hybrid.spec,
                obs_space=env_f.obs_space,
                act_space=env_f.act_space,
            ).to(device)

            wm_trainer = hybrid.Trainer(
                cfg=cfg.wm.hybrid.train,
                wm=wm,
                ctx=self,
            )

            seq_len = cfg.wm.hybrid.train.seq_len
            buf = env_f.chunk_buffer(cfg.buffer_cap, seq_len, sampler)

            if cfg.agent.type == "dreamer":
                actor = hybrid.impl.Actor(
                    cfg=cfg.wm.hybrid.spec,
                    act_space=env_f.act_space,
                ).to(device)
                agent_fn = lambda: hybrid.wm.VecAgent(wm, actor)

                critic_ctor = lambda: hybrid.impl.Critic(
                    cfg=cfg.wm.hybrid.spec,
                ).to(device)

                ac_trainer = hybrid.dreamer.Trainer(
                    cfg=cfg.agent.dreamer,
                    wm=wm,
                    act_space=env_f.act_space,
                    actor=actor,
                    critic_ctor=critic_ctor,
                    ctx=self,
                )
            elif cfg.agent.type == "cem":
                planner = hybrid.cem.CEMPlanner(
                    cfg=cfg.agent.cem,
                    wm=wm,
                    act_space=env_f.act_space,
                )
                agent_fn = lambda: hybrid.wm.VecAgent(wm, planner)
            else:
                raise NotImplementedError(cfg.agent.type)

        elif cfg.wm.type == "deter":
            wm = deter.impl.WorldModel(
                cfg=cfg.wm.deter.spec,
                obs_space=env_f.obs_space,
                act_space=env_f.act_space,
            ).to(device)

            wm_trainer = deter.Trainer(
                cfg=cfg.wm.deter.train,
                wm=wm,
                ctx=self,
            )

            seq_len = cfg.wm.deter.train.seq_len
            buf = env_f.chunk_buffer(cfg.buffer_cap, seq_len, sampler)

            if cfg.agent.type == "cem":
                planner = deter.cem.CEMPlanner(
                    cfg=cfg.agent.cem,
                    wm=wm,
                    act_space=env_f.act_space,
                )
                agent_fn = lambda: deter.wm.VecAgent(wm, planner)
            elif cfg.agent.type == "ppo":
                actor = deter.impl.Actor(
                    cfg=cfg.wm.deter.spec,
                    act_space=env_f.act_space,
                ).to(device)
                agent_fn = lambda: deter.wm.VecAgent(wm, actor)

                critic_ctor = lambda: deter.impl.Critic(
                    cfg=cfg.wm.deter.spec,
                ).to(device)

                ac_trainer = deter.ppo.Trainer(
                    cfg=cfg.agent.ppo,
                    wm=wm,
                    actor=actor,
                    act_space=env_f.act_space,
                    make_critic=critic_ctor,
                    ctx=self,
                )
            else:
                raise NotImplementedError(cfg.agent.type)

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
        ep_rets = [0 for _ in range(train_envs.num_envs)]

        def update_env_step(n: int = 1):
            nonlocal env_step
            env_step += n
            pbar.update(n)
            train_agent.eps = 1.0 - env_step / cfg.total_steps

        while True:
            if should_val:
                val_rets = []
                for _, ep in ProgressBar(
                    rollout.episodes(
                        val_envs,
                        val_agent,
                        max_episodes=cfg.val_episodes,
                    ),
                    desc="Val",
                    total=cfg.val_episodes,
                    leave=False,
                ):
                    val_rets.append(sum(ep.reward))

                self.exp.add_scalar("val/returns", np.mean(val_rets))

            if should_end:
                break

            for _ in range(cfg.env_steps):
                env_idx, step = next(train_iter)
                ep_ids[env_idx], _ = buf.push(ep_ids[env_idx], step)
                ep_rets[env_idx] += step.reward
                if step.done:
                    ep_ret = ep_rets[env_idx]
                    self.exp.add_scalar("train/ep_ret", ep_ret)
                    ep_rets[env_idx] = 0.0
                update_env_step()
                prof.update()

            for _ in range(cfg.opt_steps):
                if len(buf) < cfg.prefill:
                    continue

                if cfg.wm.type == "hybrid":
                    idxes = sampler.sample(cfg.wm.hybrid.train.batch_size)
                    batch = env_f.fetch_chunk_batch(buf, idxes)
                    states = wm_trainer.opt_step(batch)

                    if cfg.agent.type == "dreamer":
                        init_s = flat(states[1:-1])
                        n = cfg.agent.dreamer.batch_size
                        init_s = init_s[torch.randint(len(init_s), size=(n,))]
                        ac_trainer.opt_step(init_s)

                elif cfg.wm.type == "deter":
                    idxes = sampler.sample(cfg.wm.deter.train.batch_size)
                    batch = env_f.fetch_chunk_batch(buf, idxes)
                    states = wm_trainer.opt_step(batch)

                    if cfg.agent.type == "ppo":
                        init_s = flat(states[1:-1])
                        n = cfg.agent.ppo.batch_size
                        init_s = init_s[torch.randint(len(init_s), size=(n,))]
                        ac_trainer.opt_step(init_s)

                prof.update()


def main():
    dicts = []
    presets = ["CartPole-v1"]
    # presets = ["LunarLander-v2"]
    # presets = ["HalfCheetah-v4"]

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
