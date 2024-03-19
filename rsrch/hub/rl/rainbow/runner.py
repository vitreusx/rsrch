import argparse
import logging
import pickle
import queue
import threading
import time
from collections import defaultdict
from copy import copy
from datetime import datetime
from functools import partial
from itertools import islice
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Lock, Thread
from typing import Type, TypeVar

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn

from rsrch import _exp as exp
from rsrch import spaces
from rsrch.nn import noisy
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.rl.utils import polyak
from rsrch.utils import _sched as sched
from rsrch.utils import cron, repro
from rsrch.utils.parallel import Manager

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


class QAgent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, q: nets.Q, q_lock, val=False):
        nn.Module.__init__(self)
        self.q, self.q_lock = q, q_lock
        self.env_f = env_f
        self.val = val

    def policy(self, obs: np.ndarray):
        obs = self.env_f.move_obs(obs)
        with self.q_lock:
            q = self.q(obs, val_mode=self.val)
        if isinstance(q, ValueDist):
            q = q.mean
        act = q.argmax(-1)
        return self.env_f.move_act(act, to="env")


class DataWorker:
    def __init__(
        self,
        cfg: config.Config,
        q: nets.Q,
        q_lock,
        batch_queue: mp.Queue,
        device: torch.device,
    ):
        self.cfg = cfg
        self.batch_queue = batch_queue
        self.q_lock = q_lock

        self.device = device
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
        self.buf_lock = threading.Lock()

        self.batch_size = self.cfg.opt.batch_size
        self.fetch_thr = Thread(target=self._fetch_run)

        self.envs = self.env_f.vector_env(self.cfg.num_envs, mode="train")
        self.expl_eps = 0.0
        self.agent = gym.vector.agents.EpsAgent(
            opt=QAgent(self.env_f, q, q_lock, val=False),
            rand=gym.vector.agents.RandomAgent(self.envs),
            eps=self.expl_eps,
        )

        self.env_iter = rollout.steps(self.envs, self.agent)
        self.ep_ids = defaultdict(lambda: None)
        self.ep_rets = defaultdict(lambda: 0.0)

    def fetch_start(self):
        self.fetch_thr.start()

    def _fetch_run(self):
        while True:
            payload = self.fetch_batch()
            self.batch_queue.put(payload)

    def fetch_batch(self):
        with self.buf_lock:
            if isinstance(self.sampler, data.PrioritizedSampler):
                idxes, is_coefs = self.sampler.sample(self.batch_size)
                batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
                return idxes, is_coefs, batch
            else:
                idxes = self.sampler.sample(self.batch_size)
                batch = self.env_f.fetch_slice_batch(self.env_buf, idxes)
                return idxes, batch

    def step_async(self):
        self.env_iter.step_async()

    def step_wait(self):
        ep_rets = {}
        for env_idx, step in self.env_iter.step_wait():
            with self.buf_lock:
                self.ep_ids[env_idx], slice_id = self.env_buf.push(
                    self.ep_ids[env_idx], step
                )
            self.ep_rets[env_idx] += step.reward

            if isinstance(self.sampler, data.PrioritizedSampler):
                if slice_id is not None:
                    with self.buf_lock:
                        max_prio = self.sampler._max.total
                        if max_prio == 0.0:
                            max_prio = 1.0
                        self.sampler[slice_id] = max_prio

            if step.done:
                del self.ep_ids[env_idx]
                ep_rets[env_idx] = self.ep_rets[env_idx]
                del self.ep_rets[env_idx]
            else:
                ep_rets[env_idx] = None

        return ep_rets

    def update_prio(self, idxes, prio, prio_exp):
        with self.buf_lock:
            prio = prio.float().cpu().numpy() ** prio_exp
            self.sampler.update(idxes, prio)


class Runner:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    def train(self):
        self.prepare()
        self.do_train_loop()

    def prepare(self):
        cfg = self.cfg

        env_id = getattr(cfg.env, cfg.env.type).env_id
        self.exp = exp.Experiment(
            project="rainbow",
            run=f"{env_id}__{datetime.now():%Y-%m-%d_%H-%M-%S}",
            board=exp.board.Tensorboard,
            requires=cfg.requires,
        )

        self.rng = repro.RandomState()
        self.rng.init(
            seed=cfg.random.seed,
            deterministic=cfg.random.deterministic,
        )

        # torch.autograd.set_detect_anomaly(True)

        self.env_step, self.opt_step, self.agent_step = 0, 0, 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.exp.register_step("opt_step", lambda: self.opt_step)
        self.exp.register_step("agent_step", lambda: self.agent_step)

        self.device = self.exp.exec_env.device

        env_f = env.make_factory(cfg.env, self.device, seed=cfg.random.seed)
        assert isinstance(env_f.obs_space, spaces.torch.Image)
        assert isinstance(env_f.act_space, spaces.torch.Discrete)
        self._frame_skip = getattr(env_f, "frame_skip", 1)

        def make_qf():
            qf = nets.Q(cfg, env_f.obs_space, env_f.act_space)
            qf = qf.to(self.device)
            qf = qf.share_memory()
            return qf

        self.qf, self.qf_t = make_qf(), make_qf()
        self.qf_opt = cfg.opt.optimizer(self.qf.parameters())
        self.should_update_q = self._make_every(self.cfg.nets.polyak)

        if cfg.data.parallel:
            m = Manager(mp.get_context("spawn"))
            self.q_lock = m.ctx.Lock()
            q_ref = m.local_ref(self.qf)
            batch_queue = m.ctx.Queue(maxsize=cfg.data.prefetch_factor)
            self.data = m.remote(DataWorker)(
                cfg, q_ref, self.q_lock, batch_queue, self.device
            )
        else:
            self.q_lock = NoLock()
            self.data = DataWorker(cfg, self.qf, self.q_lock, None, self.device)

        self.val_envs = env_f.vector_env(cfg.val.envs, mode="val")
        self.val_agent = QAgent(env_f, self.qf, self.q_lock, val=True)

        amp_enabled = cfg.opt.dtype != "float32"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.autocast = lambda: torch.autocast(
            device_type=self.device.type,
            dtype=getattr(torch, cfg.opt.dtype),
            enabled=amp_enabled,
        )

        gammas = torch.tensor([cfg.gamma**i for i in range(cfg.data.slice_len)])
        self.gammas = gammas.to(self.device)
        self.final_gamma = cfg.gamma**cfg.data.slice_len

        self.agent_eps = self._make_sched(cfg.expl.eps)
        self.prio_exp = self._make_sched(cfg.prioritized.prio_exp)
        self.is_coef_exp = self._make_sched(cfg.prioritized.is_coef_exp)

        if cfg.resume is not None:
            self.load(cfg.resume)

    def _make_every(self, cfg: dict):
        if isinstance(cfg["every"], dict):
            every, unit = cfg["every"]["n"], cfg["every"]["of"]
        else:
            every, unit = cfg["every"], "env_step"

        return cron.Every2(
            step_fn=lambda: getattr(self, unit),
            every=every,
            iters=cfg.get("iters", 1),
            never=cfg.get("never", False),
        )

    def _make_until(self, cfg):
        if isinstance(cfg, dict):
            max_value, unit = cfg["n"], cfg["of"]
        else:
            max_value, unit = cfg, "env_step"

        return cron.Until(
            step_fn=lambda: getattr(self, unit),
            max_value=max_value,
        )

    def _make_sched(self, cfg):
        if isinstance(cfg, dict):
            desc, unit = cfg["desc"], cfg["of"]
        else:
            desc, unit = cfg, "env_step"
        return sched.Auto(desc, lambda: getattr(self, unit))

    def _make_pbar(self, until: cron.Until, *args, **kwargs):
        return self.exp.pbar(
            *args,
            **kwargs,
            total=until.max_value,
            initial=until.step_fn(),
        )

    def do_train_loop(self):
        self.should_log = self._make_every(self.cfg.log)

        should_warmup = self._make_until(self.cfg.warmup)
        with self._make_pbar(should_warmup, desc="Warmup") as self.pbar:
            while should_warmup:
                self.data.step_async()
                self._step_wait()

        should_val = self._make_every(self.cfg.val.sched)
        should_opt = self._make_every(self.cfg.opt.sched)

        should_train = self._make_until(self.cfg.total)
        with self._make_pbar(should_train, desc="Train") as self.pbar:
            while should_train:
                if should_val:
                    self._val_epoch()
                self.data.step_async()
                while should_opt:
                    self._opt_step()
                self._step_wait()

    def _val_epoch(self):
        val_iter = self.exp.pbar(
            islice(self._val_ret_iter(), 0, self.cfg.val.episodes),
            desc="Val",
            total=self.cfg.val.episodes,
            leave=False,
        )

        with self.q_lock:
            noisy.zero_noise_(self.qf)
            self.exp.add_scalar("val/mean_ret", np.mean([*val_iter]))

    def _opt_step(self):
        if hasattr(self, "batch_queue"):
            payload = self.batch_queue.get()
        else:
            payload = self.data.fetch_batch()

        if self.cfg.prioritized.enabled:
            idxes, is_coefs, batch = payload
        else:
            idxes, batch = payload

        if self.cfg.aug.rew_clip is not None:
            batch.reward.clamp_(*self.cfg.aug.rew_clip)

        with torch.no_grad():
            with self.autocast():
                next_q_eval = self.qf_t(batch.obs[-1])

                if isinstance(next_q_eval, ValueDist):
                    if self.cfg.double_dqn:
                        with self.q_lock:
                            next_q_act: ValueDist = self.qf(batch.obs[-1])
                    else:
                        next_q_act = next_q_eval
                    act = next_q_act.mean.argmax(-1)
                    target = next_q_eval.gather(-1, act[..., None])
                    target = target.squeeze(-1)
                elif isinstance(next_q_eval, Tensor):
                    if self.cfg.double_dqn:
                        with self.q_lock:
                            next_q_act: Tensor = self.qf(batch.obs[-1])
                        act = next_q_act.argmax(-1)
                        target = next_q_eval.gather(-1, act[..., None])
                        target = target.squeeze(-1)
                    else:
                        target = next_q_eval.max(-1).values

                gamma_t = (1.0 - batch.term.float()) * self.final_gamma
                returns = (batch.reward * self.gammas.unsqueeze(-1)).sum(0)
                target = returns + gamma_t * target

        with self.autocast():
            with self.q_lock:
                qv = self.qf(batch.obs[0])
            pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

            if isinstance(target, ValueDist):
                # prio = q_losses = ValueDist.proj_kl_div(target, pred)
                prio = q_losses = ValueDist.apx_w1_div(target, pred)
            else:
                prio = (pred - target).abs()
                q_losses = (pred - target).square()

            if self.cfg.prioritized.enabled:
                is_coefs = torch.as_tensor(is_coefs, device=self.device)
                q_losses = (is_coefs ** self.is_coef_exp()) * q_losses

            loss = q_losses.mean()

        self.qf_opt.zero_grad(set_to_none=True)

        with self.q_lock:
            self.scaler.scale(loss).backward()
            if self.cfg.opt.grad_clip is not None:
                self.scaler.unscale_(self.qf_opt)
                nn.utils.clip_grad_norm_(self.qf.parameters(), self.cfg.opt.grad_clip)
            self.scaler.step(self.qf_opt)
            self.scaler.update()

        self.opt_step += 1

        if self.should_log:
            self.exp.add_scalar("train/loss", loss)
            if isinstance(pred, ValueDist):
                self.exp.add_scalar("train/mean_q_pred", pred.mean.mean())
            else:
                self.exp.add_scalar("train/mean_q_pred", pred.mean())

        if self.cfg.prioritized.enabled:
            prio_exp = self.prio_exp()
            self.data.update_prio(idxes, prio.detach(), prio_exp)

    def _step_wait(self):
        ep_rets = self.data.step_wait()
        for env_idx, ep_ret in ep_rets.items():
            if ep_ret is not None:
                self.exp.add_scalar("train/ep_ret", ep_ret)

            self.env_step += self._frame_skip
            self.agent_step += 1
            if hasattr(self, "pbar"):
                self.pbar.n = self.env_step
                self.pbar.update(0)

    def _val_ret_iter(self):
        val_ep_rets = defaultdict(lambda: 0.0)
        for env_idx, step in rollout.steps(self.val_envs, self.val_agent):
            val_ep_rets[env_idx] += step.reward
            if step.done:
                yield val_ep_rets[env_idx]
                del val_ep_rets[env_idx]

    def _state_dict(self):
        state = {}
        state["rng"] = self.rng.save()
        for name in ("qf", "qf_t", "qf_opt"):
            state[name] = getattr(self, name).state_dict()
        return state

    def _load_state_dict(self, state: dict):
        self.rng.load(state["rng"])
        for name in ("qf", "qf_t", "qf_opt"):
            getattr(self, name).load_state_dict(state[name])

    def save(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state_dict()
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        self._load_state_dict(state)


def main():
    presets = ["faster", "der"]
    cfg = config.from_args(
        config.Config,
        argparse.Namespace(presets=presets),
        config_file=Path(__file__).parent / "config.yml",
        presets_file=Path(__file__).parent / "presets.yml",
    )

    runner = Runner(cfg)
    runner.train()
