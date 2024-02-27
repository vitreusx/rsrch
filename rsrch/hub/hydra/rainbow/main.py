import argparse
import queue
import time
from collections import defaultdict
from copy import copy
from datetime import datetime
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock, Thread
from typing import Type, TypeVar

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.profiler import profiler
from rsrch.nn import noisy
from rsrch.rl import data, gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class QAgent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, qf: nets.Q, val=False):
        nn.Module.__init__(self)
        self.qf = qf
        self.env_f = env_f
        self.val = val

    def policy(self, obs: np.ndarray):
        with infer_ctx(self.qf):
            obs = self.env_f.move_obs(obs)
            if self.val:
                noisy.zero_noise_(self.qf)
            else:
                noisy.reset_noise_(self.qf)
            q: ValueDist | Tensor = self.qf(obs)
            if isinstance(q, ValueDist):
                act = q.mean.argmax(-1)
            else:
                act = q.argmax(-1)
            return self.env_f.move_act(act, to="env")


class _Remote:
    def __init__(self, cls, *args, **kwargs):
        master, slave = mp.Pipe(duplex=True)
        self._conn = master
        self._proc = mp.Process(
            target=self.proc_target,
            args=(cls, slave, *args),
            kwargs=kwargs,
        )
        self._store, self._req_idx = {}, 0
        self._func_cache = {}

    @staticmethod
    def proc_target(cls, conn: Connection, *args, **kwargs):
        conn.recv()
        worker = cls(*args, **kwargs)
        while True:
            idx, cmd, args, kwargs = conn.recv()
            f = getattr(worker, cmd)
            if not callable(f):
                raise ValueError(f"Cannot access '{cmd}' of remote worker.")
            ret = f(*args, **kwargs)
            conn.send((idx, ret))

    def __getattr__(self, __name: str):
        if __name not in self._func_cache:

            def _func(*args, **kwargs):
                req_idx = self._req_idx
                self._conn.send((req_idx, __name, args, kwargs))
                self._req_idx += 1
                while req_idx not in self._store:
                    resp_idx, ret = self._conn.recv()
                    self._store[resp_idx] = ret
                ret = self._store[req_idx]
                del self._store[req_idx]
                return ret

            self._func_cache = _func

        return self._func_cache[__name]


T = TypeVar("T")


def remote(cls: Type[T]) -> Type[T]:
    cls = copy(cls)
    cls.remote = partial(_Remote, cls)
    return cls


class DataWorker:
    @staticmethod
    def target(rpc_conn: Connection, *args, **kwargs):
        rpc_conn.recv()  # Sync message
        worker = DataWorker(*args, **kwargs)
        while True:
            idx, cmd, args, kwargs = rpc_conn.recv()
            retval = getattr(worker, cmd)(*args, **kwargs)
            rpc_conn.send((idx, retval))

    def __init__(self, cfg: config.Config, qf: nets.Q, batch_queue):
        self.cfg = cfg
        self.qf = qf
        self.batch_queue = batch_queue

        self.device = torch.device(self.cfg.device)
        self.env_f = env.make_factory(self.cfg.env, self.device, seed=cfg.random.seed)

        if self.cfg.prioritized.enabled:
            self.sampler = data.PrioritizedSampler(max_size=self.cfg.data.buf_cap)
        else:
            self.sampler = data.UniformSampler()

        self.env_buf = self.env_f.slice_buffer(
            self.cfg.data.buf_cap,
            self.cfg.data.slice_len,
            sampler=self.sampler,
        )
        self.mtx = Lock()

        self.batch_size = self.cfg.opt.batch_size
        self.fetch_thr = Thread(target=self._fetch_run)

        self.envs = self.env_f.vector_env(self.cfg.num_envs, mode="train")
        self.env_step = 0
        self.agent = gym.vector.agents.EpsAgent(
            opt=QAgent(self.env_f, self.qf),
            rand=gym.vector.agents.RandomAgent(self.envs),
            eps=self._agent_eps(),
        )

        self.env_iter = iter(rollout.steps(self.envs, self.agent))
        self.ep_ids = defaultdict(lambda: None)
        self.ep_rets = defaultdict(lambda: 0.0)

    def _agent_eps(self):
        if self.env_step < self.cfg.warmup:
            return 1.0
        else:
            return self.cfg.expl.eps(self.env_step)

    def fetch_start(self):
        self.fetch_thr.start()

    def _fetch_run(self):
        while True:
            payload = self.fetch_batch()
            self.batch_queue.put(payload)

    def fetch_batch(self):
        with self.mtx:
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
        with self.mtx:
            self.ep_ids[env_idx], slice_id = self.env_buf.push(
                self.ep_ids[env_idx], step
            )
        self.ep_rets[env_idx] += step.reward

        if isinstance(self.sampler, data.PrioritizedSampler):
            if slice_id is not None:
                with self.mtx:
                    max_prio = self.sampler._max.total
                    if max_prio == 0.0:
                        max_prio = 1.0
                    self.sampler[slice_id] = max_prio

        ep_rets = None
        if step.done:
            del self.ep_ids[env_idx]
            ep_rets = self.ep_rets[env_idx]
            del self.ep_rets[env_idx]

        self.env_step += self.env_f.frame_skip
        self.agent.eps = self._agent_eps()
        return ep_rets

    def update_prio(self, idxes, prio):
        with self.mtx:
            prio_exp = self.cfg.prioritized.prio_exp(self.env_step)
            prio = prio.float().detach().cpu().numpy() ** prio_exp
            self.sampler.update(idxes, prio)


class RPC:
    def __init__(self, worker: mp.Process | Thread, conn: Connection):
        self.worker = worker
        self.conn = conn
        worker.start()
        self._cache = {}

    def __getattr__(self, name):
        if name not in self._cache:

            def _func(*args, **kwargs):
                self.conn.send((None, name, args, kwargs))
                _, retval = self.conn.recv()
                return retval

            self._cache[name] = _func

        return self._cache[name]


def main():
    presets = ["faster"]
    # presets = ["faster", "der"]

    cfg = config.from_args(
        config.Config,
        argparse.Namespace(presets=presets),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    rng = repro.RandomState()
    rng.init(cfg.random.seed, benchmark=cfg.random.benchmark)

    opt_step, env_step, agent_step = 0, 0, 0
    start_time = time.perf_counter()

    def make_sched(cfg):
        every, unit = cfg["every"]
        step_fn = {
            "opt_step": lambda: opt_step,
            "env_step": lambda: env_step,
            "agent_step": lambda: agent_step,
            "time": lambda: time.perf_counter() - start_time,
        }[unit]
        return cron.Every2(
            step_fn=step_fn,
            every=every,
            iters=cfg.get("iters", 1),
            never=cfg.get("never", False),
        )

    device = torch.device(cfg.device)
    mp.set_start_method("spawn", force=True)

    env_id = getattr(cfg.env, cfg.env.type).env_id
    exp = tensorboard.Experiment(
        project="rainbow",
        run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    prof = profiler(exp.dir / "traces", device)
    prof.register("env_step", "opt_step")

    env_f = env.make_factory(cfg.env, device, seed=cfg.random.seed)
    assert isinstance(env_f.obs_space, spaces.torch.Image)
    assert isinstance(env_f.act_space, spaces.torch.Discrete)

    def make_qf():
        qf = nets.Q(cfg, env_f.obs_space, env_f.act_space)
        if cfg.expl.noisy:
            noisy.replace_(qf, cfg.expl.sigma0)
        qf = qf.to(device)
        qf = qf.share_memory()
        return qf

    qf, qf_t = make_qf(), make_qf()

    if cfg.data.parallel:
        batch_queue = mp.Queue(maxsize=cfg.data.prefetch_factor)
        master, slave = mp.Pipe(duplex=True)
        args = (cfg, qf, batch_queue)
        proc = mp.Process(target=DataWorker.target, args=(slave, *args))
        data_: DataWorker = RPC(proc, master)
        # For some reason, the network's parameters appear as though they were
        # zero-initialized in the child process. This can be "fixed" by
        # reinitializing the network.
        polyak.sync(make_qf(), qf)
        # Send the sync message
        master.send(())
    else:
        batch_queue = queue.Queue(maxsize=cfg.data.prefetch_factor)
        args = (cfg, qf, batch_queue)
        data_ = DataWorker(*args)

    qf_opt = cfg.opt.optimizer(qf.parameters())

    polyak.sync(qf, qf_t)
    # qf_polyak = polyak.Polyak(qf, qf_t, **cfg.nets.polyak)
    should_update = make_sched(cfg.nets.polyak)

    env_step, opt_step = 0, 0
    exp.register_step("env_step", lambda: env_step, default=True)

    def take_env_step():
        ep_ret = data_.take_env_step()

        if ep_ret is not None:
            exp.add_scalar("train/ep_ret", ep_ret)

        nonlocal env_step, agent_step
        env_step += env_f.frame_skip
        agent_step += 1
        pbar.update(env_f.frame_skip)

    val_envs = env_f.vector_env(cfg.num_envs, mode="val")
    val_agent = QAgent(env_f, qf, val=True)
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

    should_val = make_sched(cfg.val.sched)
    should_opt = make_sched(cfg.opt.sched)
    should_log = make_sched(cfg.log)

    gammas = torch.tensor([cfg.gamma**i for i in range(cfg.data.slice_len)])
    gammas = gammas.to(device)
    final_gamma = cfg.gamma**cfg.data.slice_len

    pbar = tqdm(desc="Warmup", total=cfg.warmup, initial=env_step)
    while env_step < cfg.warmup:
        take_env_step()

    data_.fetch_start()

    pbar = tqdm(desc="Train", total=cfg.total_env_steps, initial=env_step)
    while env_step < cfg.total_env_steps:
        if should_val:
            val_iter = tqdm(
                make_val_iter(),
                desc="Val",
                total=cfg.val.episodes,
                leave=False,
            )

            noisy.zero_noise_(qf)
            rets = [sum(ep.reward) for _, ep in val_iter]
            exp.add_scalar("val/mean_ret", np.mean(rets))

        with prof.region("env_step"):
            take_env_step()

        while should_update:
            polyak.update(qf, qf_t, tau=cfg.nets.polyak["tau"])

        # Opt step
        while should_opt:
            with prof.region("opt_step"):
                if cfg.prioritized.enabled:
                    idxes, is_coefs, batch = batch_queue.get()
                else:
                    idxes, batch = batch_queue.get()

                if cfg.aug.rew_clip is not None:
                    batch.reward.clamp_(*cfg.aug.rew_clip)

                with torch.no_grad():
                    with autocast():
                        noisy.reset_noise_(qf_t)
                        next_q_eval = qf_t(batch.obs[-1])

                        if isinstance(next_q_eval, ValueDist):
                            if cfg.double_dqn:
                                noisy.reset_noise_(qf)
                                next_q_act: ValueDist = qf(batch.obs[-1])
                            else:
                                next_q_act = next_q_eval
                            act = next_q_act.mean.argmax(-1)
                            target = next_q_eval.gather(-1, act[..., None])
                            target = target.squeeze(-1)
                        elif isinstance(next_q_eval, Tensor):
                            if cfg.double_dqn:
                                noisy.reset_noise_(qf)
                                next_q_act: Tensor = qf(batch.obs[-1])
                                act = next_q_act.argmax(-1)
                                target = next_q_eval.gather(-1, act[..., None])
                                target = target.squeeze(-1)
                            else:
                                target = next_q_eval.max(-1).values

                        gamma_t = (1.0 - batch.term.float()) * final_gamma
                        returns = (batch.reward * gammas.unsqueeze(-1)).sum(0)
                        target = returns + gamma_t * target

                with autocast():
                    noisy.reset_noise_(qf)
                    qv = qf(batch.obs[0])
                    pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

                    if isinstance(target, ValueDist):
                        prio = q_losses = ValueDist.proj_kl_div(target, pred)
                    else:
                        prio = (pred - target).abs()
                        q_losses = (pred - target).square()

                    if cfg.prioritized.enabled:
                        is_coef_exp = cfg.prioritized.is_coef_exp(env_step)
                        is_coefs = torch.as_tensor(is_coefs, device=device)
                        q_losses = (is_coefs**is_coef_exp) * q_losses

                    loss = q_losses.mean()

                qf_opt.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()
                if cfg.opt.grad_clip is not None:
                    scaler.unscale_(qf_opt)
                    nn.utils.clip_grad_norm_(qf.parameters(), cfg.opt.grad_clip)
                scaler.step(qf_opt)
                scaler.update()

                opt_step += 1

                if should_log:
                    exp.add_scalar("train/loss", loss)
                    if isinstance(pred, ValueDist):
                        exp.add_scalar("train/mean_q_pred", pred.mean.mean())
                    else:
                        exp.add_scalar("train/mean_q_pred", pred.mean())

                if cfg.prioritized.enabled:
                    data_.update_prio(idxes, prio)
