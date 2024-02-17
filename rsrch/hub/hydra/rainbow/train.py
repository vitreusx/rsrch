import argparse
import queue
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock, Thread
from typing import Generic, TypeVar

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.profiler import profiler
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class Agent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, qf: nets.Q):
        nn.Module.__init__(self)
        self.qf = qf
        self.env_f = env_f

    def policy(self, obs: np.ndarray):
        with infer_ctx(self.qf):
            obs = self.env_f.move_obs(obs)
            q: ValueDist | Tensor = self.qf(obs)
            if isinstance(q, ValueDist):
                act = q.mean.argmax(-1)
            else:
                act = q.argmax(-1)
            return self.env_f.move_act(act, to="env")


class EnvProc(mp.Process):
    def __init__(
        self,
        cfg: config.Config,
        qf: nets.Q,
        batch_queue: mp.Queue,
        comm: Connection,
    ):
        super().__init__()
        self.cfg = cfg
        self.qf = qf
        self.batch_queue = batch_queue
        self.comm = comm

    def run(self):
        self.init()
        while True:
            cmd, args, kwargs = self.comm.recv()
            retval = getattr(self, cmd)(*args, **kwargs)
            self.comm.send(retval)

    def init(self):
        self.device = torch.device(self.cfg.device)
        self.env_f = env.make_factory(self.cfg.env, self.device)

        if self.cfg.prioritized.enabled:
            self.sampler = data.PrioritizedSampler(max_size=self.cfg.data.buf_cap)
        else:
            self.sampler = data.UniformSampler()

        self.env_buf = self.env_f.slice_buffer(
            self.cfg.data.buf_cap,
            self.cfg.data.slice_len,
            sampler=self.sampler,
        )
        self.env_buf_mtx = Lock()

        self.batch_size = self.cfg.opt.batch_size
        self.fetch_thr = Thread(target=self._fetch_run)

        self.envs = self.env_f.vector_env(self.cfg.num_envs, mode="train")
        self.env_step = 0
        self.agent = gym.vector.agents.EpsAgent(
            opt=Agent(self.env_f, self.qf),
            rand=gym.vector.agents.RandomAgent(self.envs),
            eps=self.cfg.expl.eps(self.env_step),
        )

        self.env_iter = iter(rollout.steps(self.envs, self.agent))
        self.ep_ids = defaultdict(lambda: None)
        self.ep_rets = defaultdict(lambda: 0.0)

    def fetch_start(self):
        self.fetch_thr.start()

    def _fetch_run(self):
        while True:
            payload = self.fetch_batch()
            self.batch_queue.put(payload)

    def fetch_batch(self):
        with self.env_buf_mtx:
            if isinstance(self.sampler, data.PrioritizedSampler):
                idxes, is_coefs = self.sampler.sample(self.batch_size)
                batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
                return idxes, is_coefs, batch
            else:
                idxes = self.sampler.sample(self.batch_size)
                batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
                return idxes, batch

    def take_env_step(self):
        env_idx, step = next(self.env_iter)
        with self.env_buf_mtx:
            self.ep_ids[env_idx], slice_id = self.env_buf.push(
                self.ep_ids[env_idx], step
            )
        self.ep_rets[env_idx] += step.reward

        if isinstance(self.sampler, data.PrioritizedSampler):
            if slice_id is not None:
                if len(self.env_buf) == 0:
                    max_prio = 1.0e3
                else:
                    max_prio = self.sampler._max.total
                self.sampler[slice_id] = max_prio

        ep_rets = None
        if step.done:
            del self.ep_ids[env_idx]
            ep_rets = self.ep_rets[env_idx]
            del self.ep_rets[env_idx]

        self.env_step += 1
        self.agent.eps = self.cfg.expl.eps(self.env_step)
        return ep_rets


class Proxy:
    def __init__(self, proc: mp.Process, conn: Connection):
        self.proc = proc
        self.conn = conn
        proc.start()

    def __getattr__(self, __name):
        def _func(*args, **kwargs):
            self.conn.send((__name, args, kwargs))
            return self.conn.recv()

        return _func


def main():
    presets = ["faster"]

    cfg = config.from_args(
        config.Config,
        argparse.Namespace(presets=presets),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)
    mp.set_start_method("spawn")

    env_id = getattr(cfg.env, cfg.env.type).env_id
    exp = tensorboard.Experiment(
        project="rainbow",
        run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    prof = profiler(exp.dir / "traces", device)
    prof.register("env_step", "opt_step")

    env_f = env.make_factory(cfg.env, device)
    assert isinstance(env_f.obs_space, spaces.torch.Image)
    assert isinstance(env_f.act_space, spaces.torch.Discrete)

    def make_qf():
        qf = nets.Q(cfg, env_f.obs_space, env_f.act_space)
        qf = qf.to(device)
        qf = qf.share_memory()
        return qf

    qf, qf_t = make_qf(), make_qf()
    qf_opt = cfg.opt.optimizer(qf.parameters())

    polyak.sync(qf, qf_t)
    qf_polyak = polyak.Polyak(qf, qf_t, **cfg.nets.polyak)

    device = torch.device(cfg.device)
    env_f = env.make_factory(cfg.env, device)

    if cfg.prioritized.enabled:
        sampler = data.PrioritizedSampler(max_size=cfg.data.buf_cap)
    else:
        sampler = data.UniformSampler()

    env_buf = env_f.slice_buffer(
        cfg.data.buf_cap,
        cfg.data.slice_len,
        sampler=sampler,
    )

    train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    env_step = 0
    train_agent = gym.vector.agents.EpsAgent(
        opt=Agent(env_f, qf),
        rand=gym.vector.agents.RandomAgent(train_envs),
        eps=cfg.expl.eps(env_step),
    )

    train_iter = iter(rollout.steps(train_envs, train_agent))
    ep_ids = defaultdict(lambda: None)
    ep_rets = defaultdict(lambda: 0.0)

    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)

    def take_env_step():
        nonlocal env_step
        env_idx, step = next(train_iter)
        ep_ids[env_idx], slice_id = env_buf.push(ep_ids[env_idx], step)
        ep_rets[env_idx] += step.reward

        if step.done:
            exp.add_scalar("train/ep_ret", ep_rets[env_idx])
            del ep_ids[env_idx]
            del ep_rets[env_idx]

        if isinstance(sampler, data.PrioritizedSampler):
            if slice_id is not None:
                if len(env_buf) == 0:
                    max_prio = 1.0e3
                else:
                    max_prio = sampler._max.total
                sampler[slice_id] = max_prio

        env_step += 1
        pbar.update()
        qf_polyak.step()

    # train_envs = env_f.vector_env(cfg.num_envs, mode="train")
    # train_agent = gym.vector.agents.EpsAgent(
    #     opt=Agent(qf),
    #     rand=gym.vector.agents.RandomAgent(train_envs),
    #     eps=cfg.expl.eps(env_step),
    # )
    # train_iter = iter(rollout.steps(train_envs, train_agent))
    # ep_ids = defaultdict(lambda: None)
    # ep_rets = defaultdict(lambda: 0.0)

    val_envs = env_f.vector_env(cfg.num_envs, mode="val")
    val_agent = Agent(env_f, qf)
    make_val_iter = lambda: iter(
        rollout.episodes(
            val_envs,
            val_agent,
            max_episodes=cfg.val.episodes,
        )
    )

    amp_enabled = cfg.opt.dtype != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    autocast = lambda: torch.autocast(
        device_type=device.type,
        dtype=getattr(torch, cfg.opt.dtype),
        enabled=amp_enabled,
    )

    global tqdm
    tqdm = partial(tqdm, dynamic_ncols=True)

    should_val = cron.Every2(lambda: env_step, **cfg.val.sched)
    should_opt = cron.Every2(lambda: env_step, **cfg.opt.sched)
    should_log = cron.Every2(lambda: env_step, **cfg.log)

    gammas = torch.tensor([cfg.gamma**i for i in range(cfg.data.slice_len)])
    gammas = gammas.to(device)
    final_gamma = cfg.gamma**cfg.data.slice_len

    pbar = tqdm(desc="Warmup", total=cfg.warmup, initial=env_step)
    while env_step < cfg.warmup:
        take_env_step()

    pbar = tqdm(desc="Train", total=cfg.total_steps, initial=env_step)
    while env_step < cfg.total_steps:
        # Val epoch
        while should_val:
            val_iter = tqdm(
                make_val_iter(),
                desc="Val",
                total=cfg.val.episodes,
                leave=False,
            )

            rets = [sum(ep.reward) for _, ep in val_iter]
            exp.add_scalar("val/mean_ret", np.mean(rets))

        with prof.region("env_step"):
            # Env step
            take_env_step()

        # Opt step
        while should_opt:
            with prof.region("opt_step"):
                if isinstance(sampler, data.PrioritizedSampler):
                    idxes, is_coefs = sampler.sample(cfg.opt.batch_size)
                    batch = env_f.fetch_slice_batch(env_buf, idxes)
                else:
                    idxes = sampler.sample(cfg.opt.batch_size)
                    batch = env_f.fetch_slice_batch(env_buf, idxes)

                if cfg.aug.rew_clip is not None:
                    batch.reward.clamp_(*cfg.aug.rew_clip)

                with torch.no_grad():
                    with autocast():
                        next_q = qf_t(batch.obs[-1])
                        if isinstance(next_q, ValueDist):
                            act = next_q.mean.argmax(-1)
                            target = next_q.gather(-1, act[..., None]).squeeze(-1)
                        else:
                            target = next_q.max(-1).values
                        target = (1.0 - batch.term.float()) * target
                        returns = (batch.reward * gammas.unsqueeze(-1)).sum(0)
                        target = returns + final_gamma * target

                with autocast():
                    pred = (
                        qf(batch.obs[0]).gather(-1, batch.act[0][..., None]).squeeze(-1)
                    )
                    if isinstance(target, ValueDist):
                        prio = q_losses = ValueDist.proj_kl_div(pred, target)
                    else:
                        prio = (pred - target).abs()
                        q_losses = (pred - target).square()

                    if cfg.prioritized.enabled:
                        is_coef_exp = cfg.prioritized.is_coef_exp(env_step)
                        q_losses = (is_coefs**is_coef_exp) * q_losses

                    loss = q_losses.mean()

                qf_opt.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()
                if cfg.opt.grad_clip is not None:
                    scaler.unscale_(qf_opt)
                    nn.utils.clip_grad_norm_(qf.parameters(), cfg.opt.grad_clip)
                scaler.step(qf_opt)
                scaler.update()

                if should_log:
                    exp.add_scalar("train/loss", loss)
                    if isinstance(pred, ValueDist):
                        exp.add_scalar("train/mean_q_pred", pred.mean.mean())
                    else:
                        exp.add_scalar("train/mean_q_pred", pred.mean())

                if isinstance(sampler, data.PrioritizedSampler):
                    prio_exp = cfg.prioritized.prio_exp(env_step)
                    prio = prio.float().detach().cpu().numpy() ** prio_exp
                    sampler.update(idxes, prio)


# if __name__ == "__main__":
#     main()
