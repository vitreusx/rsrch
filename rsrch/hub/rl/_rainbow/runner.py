import argparse
import logging
import queue
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

from rsrch import spaces
from rsrch.exp import tensorboard
from rsrch.exp.logging import Logger
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.profiler import profiler
from rsrch.nn import noisy
from rsrch.rl import data, gym
from rsrch.rl.data import _rollout as rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro
from rsrch.utils.parallel import Manager

from .. import env
from ..utils import infer_ctx
from . import config, nets
from .distq import ValueDist


class QRef:
    def __init__(self, qf: nets.Q):
        self.qf = qf

    def __call__(self, obs, val):
        with infer_ctx(self.qf):
            if val:
                noisy.zero_noise_(self.qf)
            else:
                noisy.reset_noise_(self.qf)
            q: ValueDist | Tensor = self.qf(obs)
        return q


class QAgent(gym.vector.Agent, nn.Module):
    def __init__(self, env_f: env.Factory, q_ref: QRef, val=False):
        nn.Module.__init__(self)
        self.q_ref = q_ref
        self.env_f = env_f
        self.val = val

    def policy(self, obs: np.ndarray):
        obs = self.env_f.move_obs(obs)
        q = self.q_ref(obs, self.val)
        if isinstance(q, ValueDist):
            q = q.mean
        act = q.argmax(-1)
        return self.env_f.move_act(act, to="env")


class DataWorker:
    def __init__(self, cfg: config.Config, q_ref: QRef, batch_queue: mp.Queue):
        self.cfg = cfg
        self.q_ref = q_ref
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
            opt=QAgent(self.env_f, self.q_ref),
            rand=gym.vector.agents.RandomAgent(self.envs),
            eps=self._agent_eps(),
        )

        self.env_iter = rollout.steps(self.envs, self.agent)
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

    def step_async(self):
        self.env_iter.step_async()

    def step_wait(self):
        ep_rets = {}
        for env_idx, step in self.env_iter.step_wait():
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

            if step.done:
                del self.ep_ids[env_idx]
                ep_rets[env_idx] = self.ep_rets[env_idx]
                del self.ep_rets[env_idx]
            else:
                ep_rets[env_idx] = None

            self.env_step += getattr(self.env_f, "frame_skip", 1)
            self.agent.eps = self._agent_eps()
        return ep_rets

    def update_prio(self, idxes, prio):
        with self.mtx:
            prio_exp = self.cfg.prioritized.prio_exp(self.env_step)
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

        # Setup RNG
        self.rng = repro.RandomState()
        self.rng.init(
            seed=cfg.random.seed,
            deterministic=cfg.random.deterministic,
        )

        # Setup step vars
        self.opt_step, self.env_step, self.agent_step = 0, 0, 0
        self._def_unit = "env_step"

    def _make_sched(self, cfg: dict):
        if isinstance(cfg["every"], list):
            every, unit = cfg["every"]
        else:
            every, unit = cfg["every"], self._def_unit

        return cron.Every2(
            step_fn=lambda: getattr(self, unit),
            every=every,
            iters=cfg.get("iters", 1),
            never=cfg.get("never", False),
        )

    def _make_until(self, cfg):
        if isinstance(cfg, list):
            max_value, unit = cfg
        else:
            max_value, unit = cfg, self._def_unit

        return cron.Until(
            step_fn=lambda: getattr(self, unit),
            max_value=max_value,
        )

    def save(self):
        ...

    def restore(self):
        ...


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
