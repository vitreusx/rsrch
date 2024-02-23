import argparse
import queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock, Thread

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
from rsrch.utils import cron

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
        prio_exp = self.cfg.prioritized.prio_exp(self.env_step)
        prio = prio.float().detach().cpu().numpy() ** prio_exp
        self.sampler.update(idxes, prio)


class RPC:
    def __init__(self, worker: mp.Process | Thread, conn: Connection):
        self.worker = worker
        self.conn = conn
        worker.start()
        self._cache = {}
        self._req_idx, self._futures = 0, {}

    def __getattr__(self, name):
        if name not in self._cache:

            class _func:
                def __init__(f):
                    f.name = name

                def __call__(f, *args, **kwargs):
                    fut = f.future(*args, **kwargs)
                    return fut.result()

                def future(f, *args, **kwargs):
                    req_idx = self._req_idx
                    self._req_idx += 1
                    self.conn.send((req_idx, f.name, args, kwargs))

                    class _future:
                        def __init__(fut):
                            fut.req_idx = req_idx

                        def result(fut):
                            if fut.req_idx in self._cache:
                                retval = self._cache[req_idx]
                                del self._cache[req_idx]
                                return retval

                            while True:
                                resp_idx, retval = self.conn.recv()
                                if resp_idx == fut.req_idx:
                                    return retval
                                else:
                                    self._cache[resp_idx] = retval

                    return _future()

            self._cache[name] = _func()

        return self._cache[name]


def main():
    presets = ["working"]
    # presets = ["faster", "der"]

    cfg = config.from_args(
        config.Config,
        argparse.Namespace(presets=presets),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    opt_step, env_step, agent_step = 0, 0, 0

    def make_sched(cfg):
        every, unit = cfg["every"]
        step_fn = {
            "opt_step": lambda: opt_step,
            "env_step": lambda: env_step,
            "agent_step": lambda: agent_step,
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

    env_f = env.make_factory(cfg.env, device)
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
    qf_opt = cfg.opt.optimizer(qf.parameters())

    if cfg.data.parallel:
        batch_queue = mp.Queue(maxsize=cfg.data.prefetch_factor)
        master, slave = mp.Pipe(duplex=True)
        args = (cfg, qf, batch_queue)
        proc = mp.Process(target=DataWorker.target, args=(slave, *args))
        data_: DataWorker = RPC(proc, master)
        # For some reason, the network's parameters appear as though they were
        # zero-initialized in the child process. This can be fixed by reinitializing
        # the network.
        polyak.sync(make_qf(), qf)
        # Send the sync message
        master.send(())
    else:
        batch_queue = queue.Queue(maxsize=cfg.data.prefetch_factor)
        args = (cfg, qf, batch_queue)
        data_ = DataWorker(*args)

    polyak.sync(qf, qf_t)
    # qf_polyak = polyak.Polyak(qf, qf_t, **cfg.nets.polyak)
    should_update = make_sched(cfg.nets.polyak)

    env_step, opt_step = 0, 0
    exp.register_step("env_step", lambda: env_step, default=True)

    def after_env_step(ep_ret):
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
        ep_ret = data_.take_env_step()
        after_env_step(ep_ret)

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
            ep_ret = data_.take_env_step()
            after_env_step(ep_ret)
            # fut = data_.take_env_step.future()

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
                        if cfg.double_dqn:
                            noisy.reset_noise_(qf)
                            next_q = qf(batch.obs[-1])
                        else:
                            noisy.reset_noise_(qf_t)
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
                    noisy.reset_noise_(qf)
                    qv = qf(batch.obs[0])
                    pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

                    if isinstance(target, ValueDist):
                        prio = q_losses = ValueDist.proj_kl_div(pred, target)
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

        # ep_ret = fut.result()
        # after_env_step(ep_ret)
