import concurrent.futures
import inspect
import logging
import lzma
import math
import pickle
import tempfile
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from functools import partial, wraps
from itertools import islice
from pathlib import Path
from queue import Queue
from threading import Lock, RLock, Thread
from typing import Callable, Literal, ParamSpec, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from rsrch import rl, spaces
from rsrch.exp import ExpDirExists, Experiment
from rsrch.exp.board.sqlite import Sqlite
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro, sched
from rsrch.utils.cast import safe_bind, typesafe
from rsrch.utils.early_stop import EarlyStopping

from . import adaptive, agent, config, data
from . import rl as rl_
from . import wm
from .config import Config
from .rl import a2c, ppo, sac, cem
from .wm import dreamer

P = ParamSpec("P")
R = TypeVar("R")


def exec_once(method: Callable[P, R]) -> Callable[P, R]:
    """Mark the method as a one-time-only function. Calls to the function beyond the first one will be skipped."""

    name = method.__name__
    retval = None

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        nonlocal retval
        if getattr(self, f"_{name}_done", False):
            return retval
        retval = method(self, *args, **kwargs)
        setattr(self, f"_{name}_done", True)
        return retval

    return wrapper


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def main(self):
        self.setup_base()
        if self.cfg.mode == "train":
            self.setup_train()

        self.block(self.cfg.stages)

    def block(self, tasks: list):
        for task in tasks:
            if isinstance(task, str):
                func = lambda: getattr(self, task)()
            else:
                assert len(task) == 1
                name = next(iter(task.keys()))
                params = task[name]
                if params is None:
                    params = {}
                func = safe_bind(getattr(self, name), **params)

            func()

    @exec_once
    def setup_base(self):
        repro.seed_all(self.cfg.repro.seed)
        repro.set_fully_deterministic(self.cfg.repro.determinism == "full")

        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.sdk = rl.sdk.make(self.cfg.env)

        if self.cfg.wm.type == "dreamer":
            if isinstance(self.sdk.act_space, spaces.torch.Discrete):
                self.sdk = rl.sdk.wrappers.OneHotActions(self.sdk)

        self.exp = Experiment(
            project="dreamerx",
            prefix=self.sdk.id,
            run_dir=self.cfg.run.dir,
            config=asdict(self.cfg),
            interactive=self.cfg.run.interactive,
            create_commit=self.cfg.run.create_commit,
        )

        self.exp.boards += [
            Tensorboard(dir=self.exp.dir / "board"),
            Sqlite(path=self.exp.dir / "board.db", scalars=False),
        ]

        self.exp.set_as_default(self.cfg.def_step)

        self._setup_buf()

        self._setup_wm_loader()
        self._setup_wm_val_loader()
        self._setup_wm()
        self.wm_trainer = None

        self._setup_rl_loader()
        self._setup_rl_val_loader()
        self._setup_rl()
        self.rl_trainer = None

        if self.cfg.profile.enabled:
            cfg = self.cfg.profile
            self._prof = Profiler(
                device=self.device,
                traces_dir=self.exp.dir / "traces",
                schedule=cfg.schedule,
            )
            for name in cfg.functions:
                f = self._prof.profiled(getattr(self, name))
                setattr(self, name, f)

        self.envs = self.sdk.make_envs(
            self.cfg.train.num_envs,
            mode="train",
        )
        self.agent = self._make_agent(mode="train")

        self.val_envs = self.sdk.make_envs(
            self.cfg.val.num_envs,
            mode="val",
        )
        self.val_agent = self._make_agent(mode="val")

        self.env_iter = iter(self.sdk.rollout(self.envs, self.agent))
        self._env_steps = defaultdict(lambda: 0)
        self._ep_ids = defaultdict(lambda: None)

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step)
        self.agent_step = 0
        self.exp.register_step("agent_step", lambda: self.agent_step)

    def _setup_buf(self):
        self.buf = rl.data.Buffer()
        self.buf = self.sdk.wrap_buffer(self.buf)
        self.buf_mtx = RLock()

        self.train_ids, self.val_ids = set(), deque()
        self.train_ep_ids = rl.data.Sampler()
        self.val_ep_ids = rl.data.Sampler()

        class SplitHook(rl.data.Hook):
            def on_create(hook, seq_id: int):
                total = len(self.train_ids) + len(self.val_ids)
                is_val = len(self.val_ids) <= self.cfg.data.val_frac * total
                if is_val:
                    self.val_ep_ids.add(seq_id)
                    self.val_ids.appendleft(seq_id)
                else:
                    self.train_ep_ids.add(seq_id)
                    self.train_ids.add(seq_id)

            def on_delete(hook, seq_id: int):
                if seq_id in self.train_ids:
                    del self.train_ep_ids[seq_id]
                    self.train_ids.remove(seq_id)
                else:
                    del self.val_ep_ids[seq_id]
                    self.val_ids.remove(seq_id)

        self.buf = rl.data.Observable(self.buf)
        self.buf.attach(SplitHook(), replay=True)

        self.buf = rl.data.SizeLimited(self.buf, cap=self.cfg.data.capacity)

    def _loader_ctor(self, cls: Callable[P, R]):
        """Create a constructor for loader classes. The difficulty, essentially, is that we fetch some of the kwargs from the config file - these need to be cast to proper types, whereas other args are passed `manually`, and these ought not to be either overriden, or cast to indicated types (some of them are there only for convenience.)"""

        def func(from_cfg: dict, **kwargs: P.kwargs) -> R:
            from_cfg = {k: v for k, v in from_cfg.items() if k not in kwargs}
            return typesafe(partial(cls, **kwargs))(**from_cfg)

        return func

    def _setup_wm_loader(self):
        cfg = self.cfg.data.loaders
        if self.cfg.wm.loader == "dreamer_wm":
            self.wm_loader = self._loader_ctor(data.DreamerWMLoader)(
                cfg.dreamer_wm,
                buf=self.buf,
                sampler=self.train_ep_ids,
            )
            self.wm_iter = iter(self.wm_loader)

            if self.cfg.repro.determinism != "full":
                self.wm_iter = data.make_async(self.wm_iter)

    def _setup_rl_loader(self):
        cfg = self.cfg.data.loaders
        if self.cfg.rl.loader == "dreamer_rl":
            self.rl_loader = self._loader_ctor(data.DreamerRLLoader)(
                cfg.dreamer_rl,
                real_slices=self.wm_loader,
                wm=self.wm,
                device=self.device,
                compute_dtype=self.compute_dtype,
            )
            self.rl_iter = iter(self.rl_loader)

        elif self.cfg.rl.loader == "real_rl":
            sig = inspect.signature(data.RealRLLoader)
            args = sig.bind(
                buf=self.buf,
                sampler=self.train_ep_ids,
                **cfg.real_rl,
            )
            args.apply_defaults()

            sampler_type = args.arguments["sampler_type"]
            if sampler_type == "ep_ids":
                sampler = self.train_ep_ids
            else:
                sampler = rl.data.PSampler()

            self.rl_loader = self._loader_ctor(data.RealRLLoader)(
                cfg.real_rl,
                buf=self.buf,
                sampler=sampler,
            )
            self.rl_iter = iter(self.rl_loader)

            if sampler_type == "slice_pos":
                hook = rl.data.SliceView(self.rl_loader.slice_len, sampler)
                self.buf.attach(hook)

            if self.cfg.repro.determinism != "full":
                self.rl_iter = data.make_async(self.rl_iter)

        elif self.cfg.rl.loader == "on_policy":
            temp_buf = rl.data.Buffer()
            temp_buf = self.sdk.wrap_buffer(temp_buf)

            self.rl_loader = self._loader_ctor(data.OnPolicyRLLoader)(
                cfg.on_policy,
                do_env_step=self.do_env_step,
                temp_buf=temp_buf,
            )
            self.rl_iter = iter(self.rl_loader)

        elif self.cfg.rl.loader is None:
            self.rl_loader = None

    def _setup_wm(self):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer

            obs_space, act_space = self.sdk.obs_space, self.sdk.act_space
            self.wm = dreamer.WorldModel(wm_cfg, obs_space, act_space)
        else:
            self.wm = None

        if self.wm is not None:
            self.wm = self.wm.to(self.device)

    def _setup_rl(self):
        if self.rl_loader is None:
            self.rl_obs_space = self.sdk.obs_space
            self.rl_act_space = self.sdk.act_space
        else:
            self.rl_obs_space = getattr(self.rl_loader, "obs_space", self.sdk.obs_space)
            self.rl_act_space = getattr(self.rl_loader, "act_space", self.sdk.act_space)

        rl_type = self.cfg.rl.type
        if rl_type == "a2c":
            rl_cfg = self.cfg.rl.a2c
            self.actor = a2c.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
        elif rl_type == "sac":
            rl_cfg = self.cfg.rl.sac
            self.actor = sac.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
        elif rl_type == "ppo":
            rl_cfg = self.cfg.rl.ppo
            self.actor = ppo.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
        elif rl_type == "cem":
            self.actor = None
        else:
            raise ValueError(rl_type)

        if self.actor is not None:
            self.actor = self.actor.to(self.device)

        if self.cfg.rl.loader == "dreamer_rl":
            self.rl_loader.set_actor(self.actor)
            self.rl_val_loader.set_actor(self.actor)

    def _make_agent(self, mode: Literal["train", "val"]):
        if self.cfg.rl.type == "cem":
            agent_ = cem.Agent(
                self.cfg.rl.cem,
                self.wm,
                self.compute_dtype,
            )
            agent_ = wm.Agent(
                agent_,
                wm=self.wm,
                compute_dtype=self.compute_dtype,
                pass_tensors=False,
            )

        else:
            agent_ = rl_.Agent(
                self.actor,
                sample=(mode == "train"),
                compute_dtype=self.compute_dtype,
            )
            if self.cfg.rl.loader == "dreamer_rl":
                self.rl_loader: data.DreamerRLLoader
                agent_ = wm.Agent(
                    agent_,
                    wm=self.wm,
                    compute_dtype=self.compute_dtype,
                    pass_tensors=self.rl_loader.return_tensors,
                )

        noise = getattr(self.cfg, mode).agent_noise
        agent_ = agent.Agent(agent_, noise=noise, mode=mode)

        return agent_

    @typesafe
    def _get_until_params(self, spec: config.Until):
        if isinstance(spec, int):
            max_value = spec
            of = self.cfg.def_step
        else:
            max_value, of = spec.n, spec.of
        return max_value, of

    @typesafe
    def _make_until(self, spec: config.Until):
        max_value, of = self._get_until_params(spec)
        return cron.Until(lambda of=of: getattr(self, of), max_value)

    @typesafe
    def _make_every(self, spec: config.Every | None):
        if isinstance(spec, int):
            spec = {"n": spec}
        else:
            spec = vars(spec)

        if spec["n"] is None:
            return cron.Never()

        args = {**spec}
        args["period"] = args["n"]
        del args["n"]
        of = args["of"]
        if of is None:
            of = self.cfg.def_step
        del args["of"]
        args["step_fn"] = lambda of=of: getattr(self, of)

        return cron.Every(**args)

    @typesafe
    def _make_sched(self, spec: config.Sched):
        if isinstance(spec, float):
            return lambda: spec
        else:
            if isinstance(spec, str):
                sched_fn, unit = spec, self.cfg.def_step
            else:
                sched_fn, unit = spec.value, spec.of
            return sched.Auto(sched_fn, lambda: getattr(self, unit))

    @exec_once
    def setup_train(self):
        self._setup_wm_trainer()
        self._setup_rl_trainer()

        self.wm_opt_step = 0
        self.exp.register_step("wm_opt_step", lambda: self.wm_opt_step)

        self.rl_opt_step = 0
        self.exp.register_step("rl_opt_step", lambda: self.rl_opt_step)

        self.should_log_wm = cron.Every(
            lambda: self.wm_opt_step,
            period=self.cfg.run.log_every,
            iters=None,
        )
        self.should_log_rl = cron.Every(
            lambda: self.rl_opt_step,
            period=self.cfg.run.log_every,
            iters=None,
        )

    def _setup_wm_trainer(self):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer
            self.wm_trainer = dreamer.Trainer(wm_cfg, self.wm, self.compute_dtype)
        else:
            self.wm_trainer = None

    def _setup_rl_trainer(self):
        rl_type = self.cfg.rl.type
        if rl_type == "a2c":
            rl_cfg = self.cfg.rl.a2c
            self.rl_trainer = a2c.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
                make_sched=self._make_sched,
            )

        elif rl_type == "sac":
            rl_cfg = self.cfg.rl.sac
            self.rl_trainer = sac.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
                make_sched=self._make_sched,
            )

        elif rl_type == "ppo":
            rl_cfg = self.cfg.rl.ppo
            self.rl_trainer = ppo.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
            )

        elif rl_type == "cem":
            self.rl_trainer = None

        else:
            raise ValueError(rl_type)

    def _setup_wm_val_loader(self):
        if self.cfg.wm.loader == "dreamer_wm":
            self.wm_val_step_loader = self._loader_ctor(data.DreamerWMLoader)(
                self.cfg.data.loaders.dreamer_wm,
                buf=self.buf,
                sampler=self.val_ep_ids,
                augment=None,
            )
            self.wm_val_iter = iter(self.wm_val_step_loader)

            if self.cfg.repro.determinism != "full":
                self.wm_val_iter = data.make_async(self.wm_val_iter)

            cfg = self.cfg.data.loaders.dreamer_wm
            self.wm_val_epoch_loader = data.DreamerWMLoader(
                buf=self.buf,
                sampler=self.val_ids,
                batch_size=cfg["batch_size"],
                slice_len=cfg["slice_len"],
            )

    def _setup_rl_val_loader(self):
        cfg = self.cfg.data.loaders
        if self.cfg.rl.loader == "dreamer_rl":
            self.rl_val_loader = self._loader_ctor(data.DreamerRLLoader)(
                cfg.dreamer_rl,
                real_slices=self.wm_val_step_loader,
                wm=self.wm,
                device=self.device,
                compute_dtype=self.compute_dtype,
            )
            self.rl_val_iter = iter(self.rl_val_loader)
        elif self.cfg.rl.loader == "real_rl":
            self.rl_val_loader = self._loader_ctor(data.RealRLLoader)(
                cfg.real_rl,
                buf=self.buf,
                sampler=self.val_ep_ids,
            )
            self.rl_val_iter = iter(self.rl_val_loader)

    def save_buf(
        self,
        path: str | None = None,
        tag: str | None = None,
    ):
        if path is None:
            if tag is None:
                cur_step = getattr(self, self.cfg.def_step)
                tag = f"{self.cfg.def_step}={cur_step:g}"
            dst = self.exp.dir / "data" / f"samples.{tag}.pkl.lzma"
        else:
            dst = Path(path)

        dst.parent.mkdir(parents=True, exist_ok=True)

        with lzma.open(dst, "wb") as f:
            pickle.dump(self.buf.save(), f)

        self.exp.log(logging.INFO, f"Saved buffer data to: {str(dst)}")

    def load_buf(self, path: str | Path):
        path = Path(path)
        if not path.exists() and (self.exp.dir / path).exists():
            path = self.exp.dir / path

        with lzma.open(path, "rb") as f:
            self.buf.load(pickle.load(f))

        self.exp.log(logging.INFO, f"Loaded buffer data from: {str(path)}")

    def save_ckpt(
        self,
        full: bool = False,
        tag: str | None = None,
    ):
        state = {}

        if self.wm is not None:
            state["wm"] = self.wm.state_dict()

        if self.actor is not None:
            state["actor"] = self.actor.state_dict()

        if full:
            if self.wm_trainer is not None:
                state["wm_trainer"] = self.wm_trainer.save()

            if self.rl_trainer is not None:
                state["rl_trainer"] = self.rl_trainer.save()

            state["repro"] = repro.state.save()

        if tag is None:
            cur_step = getattr(self, self.cfg.def_step)
            tag = f"ckpt.{self.cfg.def_step}={cur_step:g}"

        dst = self.exp.dir / "ckpts" / f"{tag}.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "wb") as f:
            torch.save(state, f)

        self.exp.log(logging.INFO, f"Saved checkpoint to: {str(dst)}")

    def load_ckpt(
        self,
        path: str | Path,
        only: list[str] | None = None,
    ):
        path = Path(path)

        with open(path, "rb") as f:
            state = torch.load(f, map_location="cpu")

        def should_load(key: str):
            return (key in state) and (only is None or key in only)

        if should_load("wm") and self.wm is not None:
            self.wm.load_state_dict(state["wm"])

        if should_load("actor") and self.actor is not None:
            self.actor.load_state_dict(state["actor"])

        if should_load("wm_trainer") and self.wm_trainer is not None:
            self.wm_trainer.load(state["wm_trainer"])

        if should_load("rl_trainer") and self.rl_trainer is not None:
            self.rl_trainer.load(state["rl_trainer"])

        if should_load("repro"):
            repro.state.load(state["repro"])

        self.exp.log(logging.INFO, f"Loaded checkpoint from: {str(path)}")

    def env_rollout(self, until: config.Until, mode: str = "train"):
        should_do = self._make_until(until)

        pbar = self._make_pbar("Env rollout", until=until)
        while should_do:
            self.do_env_step(mode=mode)
            self._update_pbar(pbar, should_do)

    def prefill(self, until: config.Until, ensure_nonempty: bool = True):
        pbar = self._make_pbar("Prefill", until=until)
        should_do = self._make_until(until)
        while should_do:
            self.do_env_step(mode="rand")
            self._update_pbar(pbar, should_do)

        if ensure_nonempty:
            if self.cfg.wm.loader is not None:
                while self.wm_loader.empty():
                    self.do_env_step()
                    self._update_pbar(pbar, should_do)

            if self.cfg.rl.loader is not None:
                while self.rl_loader.empty():
                    self.do_env_step()
                    self._update_pbar(pbar, should_do)

    def _make_pbar(
        self,
        desc: str,
        *,
        until: config.Until | None = None,
        of: str | None = None,
        **kwargs,
    ):
        if until is not None:
            max_value, of = self._get_until_params(until)
        else:
            max_value = None

        return self.exp.pbar(
            desc=desc,
            initial=getattr(self, of),
            total=max_value,
            unit=f" {of}",
            **kwargs,
        )

    def _update_pbar(self, pbar: tqdm, until: cron.Until):
        pbar.update(until.step_fn() - pbar.n)

    def do_env_step(self, mode="train"):
        self.agent.mode = mode
        env_idx, (step, final) = next(self.env_iter)

        with self.buf_mtx:
            self._ep_ids[env_idx] = self.buf.push(self._ep_ids[env_idx], step, final)

        if final:
            del self._ep_ids[env_idx]

        if "ep_returns" in step:
            self.exp.add_scalar("train/ep_ret", step["ep_returns"])
        if "ep_length" in step:
            self.exp.add_scalar("train/ep_len", step["ep_length"])

        self.agent_step += 1

        # Some environments skip frames (each action is repeated, say, four times.) For the purposes of benchmarking and keeping the comparisons fair, we need to keep track of the "real" number of env steps. This is done by adding "total_steps" field for all steps produced by the SDKs.
        if "total_steps" in step:
            diff = step["total_steps"] - self._env_steps[env_idx]
            self.env_step += diff
            self._env_steps[env_idx] = step["total_steps"]
        else:
            self.env_step += 1

        return env_idx, (step, final)

    def train_loop(
        self,
        until: config.Until,
        tasks: list[dict | str],
    ):
        task_functions = []
        should_run = []

        _, of = self._get_until_params(until)

        for task in tasks:
            if isinstance(task, str):
                task = {task: {}}
            every = task.get("every")
            if "every" in task:
                task = {**task}
                del task["every"]

            if every is None:
                # Task should be performed once per cycle
                should_run_task = None
            else:
                if not isinstance(every, dict):
                    # If only a number is supplied, use default 'of' value.
                    every = dict(n=every, of=of)

                # Task should be performed according to user-provided schedule.
                should_run_task = self._make_every(every)

            should_run.append(should_run_task)

            name = next(iter(task.keys()))
            params = task[name]
            if params is None:
                params = {}
            func = safe_bind(getattr(self, name), **params)
            task_functions.append(func)

        should_loop = self._make_until(until)
        pbar = self._make_pbar("Train loop", until=until)

        while should_loop:
            for task_idx in range(len(tasks)):
                flag = should_run[task_idx]
                run_task = task_functions[task_idx]

                if flag is None:
                    # Run only once
                    run_task()
                else:
                    # Run for as long as we ought
                    while flag:
                        run_task()

                self._update_pbar(pbar, should_loop)

    def do_opt_step(self, n: int = 1):
        for _ in range(n):
            self.do_wm_opt_step()
            self.do_rl_opt_step()

    def do_val_step(self):
        self.do_wm_val_step()
        self.do_rl_val_step()

    def do_val_epoch(
        self,
        max_batches: int | None = None,
        num_episodes: int = 32,
        step: str | None = None,
    ):
        _step = step

        if self.wm_trainer is not None:
            wm_loss = self.do_wm_val_epoch(max_batches)
            if wm_loss is not None:
                self.exp.add_scalar("val/wm_loss", wm_loss, step="wm_opt_step")

        val_iter = iter(self.sdk.rollout(self.val_envs, self.val_agent))
        env_rets = defaultdict(lambda: 0.0)
        all_rets = []

        pbar = self.exp.pbar(desc="Val epoch", total=num_episodes)

        for env_idx, (step, final) in val_iter:
            env_rets[env_idx] += step["reward"]
            if final:
                all_rets.append(env_rets[env_idx])
                del env_rets[env_idx]
                pbar.update()
                if len(all_rets) >= num_episodes:
                    break

        self.exp.add_scalar("val/mean_ep_ret", np.mean(all_rets), step=_step)

        self.check_plasticity()

    @torch.no_grad()
    def do_wm_val_epoch(self, max_batches=None):
        if len(self.val_ids) == 0:
            return

        val_loss, val_n = 0.0, 0

        self.wm_val_epoch_loader.h_0s.clear()
        val_iter = iter(islice(self.wm_val_epoch_loader, 0, max_batches))

        while True:
            with self.buf_mtx:
                wm_batch: data.BatchWM | None = next(val_iter, None)
                if wm_batch is None:
                    break

            wm_batch = wm_batch.to(self.device)

            self.wm_trainer.eval()
            wm_output = self.wm_trainer.compute(
                seq=wm_batch.seq,
                h_0=wm_batch.h_0,
            )

            with self.buf_mtx:
                self.wm_val_step_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

            val_loss += wm_output.loss
            val_n += 1

        if isinstance(val_loss, Tensor):
            val_loss = val_loss.item()
        val_loss = val_loss / val_n
        return val_loss

    def do_wm_opt_step(self, n: int = 1):
        if n > 1:
            for _ in range(n):
                self.do_wm_opt_step()

        self.wm_trainer.train()

        with self.buf_mtx:
            wm_batch = next(self.wm_iter)
        wm_batch = wm_batch.to(self.device)

        wm_output = self.wm_trainer.compute(
            seq=wm_batch.seq,
            h_0=wm_batch.h_0,
        )
        self.wm_trainer.opt_step(wm_output.loss)

        if isinstance(self.rl_loader, data.DreamerRLLoader):
            self.rl_loader.to_recycle = (
                wm_output.states.flatten(),
                wm_batch.seq.term.flatten(),
            )

        if wm_batch.end_pos is not None:
            # Record final state values as the future h_0 values.
            with self.buf_mtx:
                self.wm_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

        if self.should_log_wm:
            for k, v in wm_output.metrics.items():
                self.exp.add_scalar(f"wm/{k}", v, step="wm_opt_step")
            self.exp.add_scalar(f"wm/env_step", self.env_step, step="wm_opt_step")

        self.wm_opt_step += 1
        return wm_output.loss

    def do_rl_opt_step(self, n: int = 1):
        if n > 1:
            for _ in range(n):
                self.do_rl_opt_step()

        self.rl_trainer.train()

        with self.buf_mtx:
            rl_batch = next(self.rl_iter)
        rl_batch = self._move_to_device(rl_batch)

        mets = self.rl_trainer.opt_step(rl_batch)

        if self.should_log_rl:
            for k, v in mets.items():
                self.exp.add_scalar(f"rl/{k}", v, step="rl_opt_step")
            self.exp.add_scalar(f"rl/env_step", self.env_step, step="rl_opt_step")

        self.rl_opt_step += 1

    def _move_to_device(self, x):
        if hasattr(x, "to"):
            return x.to(self.device)
        elif isinstance(x, list):
            return [self._move_to_device(elem) for elem in x]
        elif isinstance(x, dict):
            return {k: self._move_to_device(v) for k, v in x.items()}
        else:
            return x

    @torch.no_grad()
    def do_wm_val_step(self):
        if len(self.val_ids) == 0:
            return

        self.wm_trainer.eval()

        with self.buf_mtx:
            wm_batch = next(self.wm_val_iter)

        wm_batch = wm_batch.to(self.device)

        wm_output = self.wm_trainer.compute(
            seq=wm_batch.seq,
            h_0=wm_batch.h_0,
        )

        if wm_batch.end_pos is not None:
            with self.buf_mtx:
                self.wm_val_step_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

        self.exp.add_scalar(f"wm/val_loss", wm_output.loss, step="wm_opt_step")

    @torch.no_grad()
    def do_rl_val_step(self):
        pass

    def check_plasticity(self):
        if self.wm_trainer is not None and hasattr(self.wm_trainer, "check_plasticity"):
            with self.buf_mtx:
                wm_batch = next(self.wm_val_iter)

            wm_batch = wm_batch.to(self.device)

            self.wm_trainer.eval()
            wm_mets, wm_res = self.wm_trainer.check_plasticity(
                seq=wm_batch.seq,
                h_0=wm_batch.h_0,
            )

            self.exp.add_dict(f"plas/wm/results", wm_res, step="rl_opt_step")
            for k, v in wm_mets.items():
                self.exp.add_scalar(f"plas/wm/{k}", v, step="wm_opt_step")

        if self.rl_trainer is not None and hasattr(self.rl_trainer, "check_plasticity"):
            with self.buf_mtx:
                rl_batch = next(self.rl_val_iter)

            rl_batch = rl_batch.to(self.device)

            self.rl_trainer.eval()
            rl_mets, rl_res = self.rl_trainer.check_plasticity(rl_batch)

            self.exp.add_dict(f"plas/rl/results", rl_res, step="rl_opt_step")
            for k, v in rl_mets.items():
                self.exp.add_scalar(f"plas/rl/{k}", v, step="rl_opt_step")

    def pretrain(
        self,
        stop_criteria: dict,
        val_every: int,
        val_on_loss_improv: float | None,
        max_val_batches: int | None,
        rl_opt_freq: float,
    ):
        should_stop = typesafe(EarlyStopping)(**stop_criteria)
        max_steps = should_stop.max_steps

        should_val = cron.Every(lambda: self.wm_opt_step, val_every)

        best_val_loss = np.inf
        best_ckpt = tempfile.mktemp(suffix=".pth")

        best_train_loss, prev_loss_val = None, None

        if rl_opt_freq > 0:
            should_opt_rl = cron.Every(
                lambda: self.wm_opt_step,
                period=1.0,
                iters=rl_opt_freq,
                accumulate=True,
            )
        else:
            should_opt_rl = cron.Never()

        pbar = self._make_pbar("Pretrain", of="wm_opt_step")
        while True:
            should_val_ = bool(should_val)
            if val_on_loss_improv is not None:
                if best_train_loss is not None and (
                    prev_loss_val is None
                    or prev_loss_val / best_train_loss > 1.0 + val_on_loss_improv
                ):
                    should_val_ = True
                    prev_loss_val = best_train_loss

            if should_val_:
                val_loss = self.do_wm_val_epoch(max_val_batches)
                self.exp.add_scalar("wm/val_loss", val_loss, step="wm_opt_step")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "wm": self.wm.state_dict(),
                            "wm_trainer": self.wm_trainer.save(),
                            "actor": self.actor.state_dict(),
                            "rl_trainer": self.rl_trainer.save(),
                        },
                        best_ckpt,
                    )

                if should_stop(val_loss, self.wm_opt_step):
                    break

            train_loss = self.do_wm_opt_step()
            if val_on_loss_improv is not None:
                if best_train_loss is None or train_loss < best_train_loss:
                    best_train_loss = train_loss

            while should_opt_rl:
                self.do_rl_opt_step()

            pbar.update()

            if max_steps is not None:
                if self.wm_opt_step >= max_steps:
                    break

        with open(best_ckpt, "rb") as f:
            best_state = torch.load(f, map_location="cpu")
            self.wm.load_state_dict(best_state["wm"])
            self.wm_trainer.load(best_state["wm_trainer"])
            self.actor.load_state_dict(best_state["actor"])
            self.rl_trainer.load(best_state["rl_trainer"])

        Path(best_ckpt).unlink()

    def pretrain_rl(self, opt_steps: int):
        self.do_rl_opt_step(opt_steps)

    def reset_wm(
        self,
        match_params: str | list[str],
        shrink_coef: float,
        pretrain: dict | None = None,
    ):
        if isinstance(match_params, list):
            match_params = "|".join(f"({pat})" for pat in match_params)

        if self.cfg.wm.type == "dreamer":
            self.wm_trainer: dreamer.Trainer

            def make_wm():
                wm = dreamer.WorldModel(
                    cfg=self.cfg.wm.dreamer,
                    obs_space=self.sdk.obs_space,
                    act_space=self.sdk.act_space,
                )
                wm = wm.to(self.device)
                return wm

            self.wm_trainer.reset(
                make_wm=make_wm,
                match_params=match_params,
                shrink_coef=shrink_coef,
            )
        else:
            raise ValueError(self.cfg.wm.type)

        if pretrain is not None:
            typesafe(self.pretrain_wm)(**pretrain)

    def reset_rl(
        self,
        match_params: str | list[str],
        shrink_coef: float,
        pretrain: dict | None = None,
    ):
        if isinstance(match_params, list):
            match_params = "|".join(f"({pat})" for pat in match_params)

        if self.cfg.rl.type == "a2c":
            self.rl_trainer: a2c.Trainer

            def make_actor():
                rl_cfg = self.cfg.rl.a2c
                actor = a2c.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
                actor = actor.to(self.device)
                return actor

            self.rl_trainer.reset(
                make_actor=make_actor,
                match_params=match_params,
                shrink_coef=shrink_coef,
            )

        else:
            raise ValueError(self.cfg.rl.type)

        if pretrain is not None:
            typesafe(self.pretrain_rl)(**pretrain)

    def adaptive_setup(
        self,
        rl_to_wm_ratio: float,
        type: Literal["v1", "v2", "v3", "v4"],
        **kwargs,
    ):
        cls = {
            "v1": adaptive.V1,
            "v2": adaptive.V2,
            "v3": adaptive.V3,
            "v4": adaptive.V4,
        }[type]
        kw = kwargs.get(type, {})
        self.wm_ratio_search = typesafe(cls)(**kw)

        self.should_opt_wm = cron.Every(
            lambda: self.env_step,
            period=self.wm_ratio_search.value,
            accumulate=True,
        )

        if rl_to_wm_ratio > 0:
            self.should_opt_rl = cron.Every(
                lambda: self.wm_opt_step,
                period=rl_to_wm_ratio,
                accumulate=True,
            )
        else:
            self.should_opt_rl = cron.Never()

    def update_adaptive_opt(self):
        val_loss = self.do_wm_val_epoch(max_batches=128)
        self.exp.add_scalar("ada/val_loss", val_loss)
        metrics = self.wm_ratio_search.update(val_loss, self.env_step)
        self.should_opt_wm.period = self.wm_ratio_search.value
        self.exp.add_scalar("ada/wm_ratio", self.should_opt_wm.period)
        for k, v in metrics.items():
            self.exp.add_scalar(f"ada/{k}", v)

    def do_adaptive_opt_step(self):
        while self.should_opt_wm:
            self.do_wm_opt_step()
            while self.should_opt_rl:
                self.do_rl_opt_step()
