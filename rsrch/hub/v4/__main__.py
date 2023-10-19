from functools import partial

import numpy as np
from . import config, env, wm, actor
import torch
from rsrch.rl import data
from rsrch.rl.data import rollout
from pathlib import Path
from rsrch.utils import cron
from rsrch.exp.profiler import Profiler
from rsrch.exp import Experiment
from tqdm.auto import tqdm
import torch.multiprocessing as mp


def main():
    cfg_d = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(cfg_d, config.Config)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    mp.set_start_method("spawn")

    loader = env.Loader(cfg.env)

    sampler = data.PrioritizedSampler(
        max_size=cfg.buffer.capacity,
    )

    buffer = loader.make_chunk_buffer(
        capacity=cfg.buffer.capacity,
        num_steps=cfg.seq_len,
        sampler=sampler,
    )

    if cfg.wm.type == "rssm":
        cfg_ = cfg.wm.rssm
        wm_ = wm.rssm.WorldModel(cfg_, loader.obs_space, loader.act_space)
        wm_ = wm_.to(device)
        obs_pred_ = wm.rssm.ObsPred(cfg_, loader.obs_space)
        obs_pred_ = obs_pred_.to(device)
        wm_trainer = wm.rssm.Trainer(cfg_, wm_, obs_pred_)
    elif cfg.wm.type == "exact":
        mod = wm.exact.from_id(cfg.env.id)
        wm_ = mod.WorldModel()
        wm_trainer = wm.exact.Trainer(wm_)
    else:
        raise NotImplementedError(cfg.wm.type)

    if cfg.actor.type == "sac":
        if cfg.wm.type == "rssm":
            cfg_ = cfg.wm.rssm
            actor_ = wm.rssm.Actor(cfg_, loader.act_space)
            actor_ = actor_.to(device)
            critic_fn = lambda: wm.rssm.Critic(cfg_).to(device)
            VecAgent = partial(wm.rssm.VecEnvAgent, wm=wm_, actor=actor_, device=device)
        elif cfg.wm.type == "exact":
            mod = wm.exact.from_id(cfg.env.id)
            actor_ = mod.Actor().to(device)
            critic_fn = lambda: mod.Critic().to(device)
            VecAgent = partial(wm.exact.Agent, wm=wm_, actor=actor_, device=device)

        ac_trainer = actor.sac.Trainer(
            cfg.actor.sac, actor_, critic_fn, wm_, loader.act_space
        )
    else:
        raise NotImplementedError(cfg.actor.type)

    num_val_envs = min(cfg.env_workers, cfg.val_episodes)
    val_envs = loader.val_envs(num_val_envs)
    val_agent = loader.VecAgent(VecAgent(num_envs=val_envs.num_envs))

    exp_envs = loader.exp_envs(cfg.exp_envs)
    exp_agent = loader.VecAgent(VecAgent(num_envs=exp_envs.num_envs))
    ep_ids = [None for _ in range(exp_envs.num_envs)]
    exp_iter = iter(rollout.steps(exp_envs, exp_agent))

    exp = Experiment(project="dreamer", config=cfg_d)
    board = exp.board
    env_step = 0
    board.add_step("env_step", lambda: env_step, default=True)
    pbar = tqdm(total=cfg.total_steps, dynamic_ncols=True)

    prof = Profiler(
        cfg=cfg.profiler,
        device=device,
        step_fn=lambda: env_step,
        trace_path=exp.dir / "trace.json",
    )

    should_val = cron.Every(lambda: env_step, cfg.val_every)
    should_stop = cron.Once(lambda: env_step >= cfg.total_steps)

    class ctx:
        should_log = cron.Every(lambda: env_step, cfg.log_every)
        board = exp.board

    def val_epoch():
        val_iter = rollout.episodes(val_envs, val_agent, max_episodes=cfg.val_episodes)
        val_returns = [sum(ep.reward) for _, ep in val_iter]
        board.add_scalar("val/returns", np.mean(val_returns))

    def collect_exp():
        nonlocal env_step
        for _ in range(cfg.exp_steps):
            env_idx, step = next(exp_iter)
            ep_ids[env_idx], chunk_id = buffer.push(ep_ids[env_idx], step)
            if chunk_id is not None:
                done = step.term or step.trunc
                prio = 1.0 if not done else cfg.seq_len
                sampler.update(chunk_id, prio)
            env_step += 1
            pbar.update()

    def opt_step():
        idxes, _ = sampler.sample(cfg.batch_size)
        batch = loader.fetch_chunk_batch(buffer, idxes)
        batch = batch.to(device)
        # if env_step < 15e3:
        #     wm_trainer.opt_step(batch, ctx)
        # else:
        #     states = wm_trainer.fetch_step(batch, ctx)
        #     ac_trainer.opt_step(states, ctx)
        states = wm_trainer.opt_step(batch, ctx)
        ac_trainer.opt_step(states, ctx)

    while True:
        if should_val:
            val_epoch()

        if should_stop:
            break

        collect_exp()
        if len(buffer) > cfg.buffer.prefill:
            opt_step()

        prof.update()


if __name__ == "__main__":
    main()
