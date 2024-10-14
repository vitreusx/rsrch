import concurrent.futures
import inspect
import logging
import lzma
import pickle
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import wraps
from itertools import islice
from pathlib import Path
from queue import Queue
from threading import Lock, RLock, Thread
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from rsrch import rl, spaces
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro, sched
from rsrch.utils.cast import argcast, safe_bind
from rsrch.utils.early_stop import EarlyStopping

from . import agent, config, data
from . import rl as rl_
from . import wm
from .config import Config
from .rl import a2c, ppo, sac
from .wm import dreamer


def exec_once(method):
    name = method.__name__

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, f"{name}_done", False):
            return
        method(self, *args, **kwargs)
        setattr(self, f"{name}_done", True)

    return wrapper


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def main(self):
        self._setup_common()

        for stage in self.cfg.stages:
            if isinstance(stage, str):
                func = lambda: getattr(self, stage)()
            else:
                assert len(stage) == 1
                name = next(iter(stage.keys()))
                params = stage[name]
                if params is None:
                    params = {}
                func = safe_bind(getattr(self, name), **params)

            func()

    @exec_once
    def _setup_common(self):
        repro.seed_all(self.cfg.repro.seed)
        repro.set_fully_deterministic(self.cfg.repro.determinism == "full")

        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.sdk = rl.sdk.make(self.cfg.env)

        self.exp = Experiment(
            project="dreamer",
            prefix=self.sdk.id,
            run_dir=self.cfg.run.dir,
            config=asdict(self.cfg),
            interactive=self.cfg.run.interactive,
            create_commit=self.cfg.run.create_commit,
        )

        board = Tensorboard(dir=self.exp.dir / "board")
        self.exp.boards.append(board)
        self.exp.set_as_default(self.cfg.def_step)

        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer

            if isinstance(self.sdk.act_space, spaces.torch.Discrete):
                self.sdk = rl.sdk.wrappers.OneHotActions(self.sdk)

            obs_space, act_space = self.sdk.obs_space, self.sdk.act_space
            self.wm = dreamer.WorldModel(wm_cfg, obs_space, act_space)
        else:
            self.wm = None

        if self.wm is not None:
            self.wm = self.wm.to(self.device)

        if self.cfg.rl.loader == "dream_rl":
            self.rl_obs_space = self.wm.state_space
            self.rl_act_space = self.wm.act_space
        else:
            self.rl_obs_space = self.sdk.obs_space
            self.rl_act_space = self.sdk.act_space

        rl_type = self.cfg.rl.type
        if rl_type == "a2c":
            rl_cfg = self.cfg.rl.a2c
            self.actor = a2c.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
        elif rl_type == "sac":
            rl_cfg = self.cfg.rl.sac
            self.actor = sac.Actor(rl_cfg.actor, self.rl_obs_space, self.rl_act_space)
        elif rl_type == "ppo":
            rl_cfg = self.cfg.rl.ppo
            self.actor = ppo.Actor(rl_cfg, self.rl_obs_space, self.rl_act_space)
        else:
            raise ValueError(rl_type)

        self.actor = self.actor.to(self.device)

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

        self.buf = rl.data.Buffer()
        self.buf = self.sdk.wrap_buffer(self.buf)
        self.buf_mtx = RLock()

        self.envs = self.sdk.make_envs(
            self.cfg.train.num_envs,
            mode="train",
        )

        self.agent = self._make_agent(mode="train")

        self.env_iter = iter(self.sdk.rollout(self.envs, self.agent))
        self._env_steps = defaultdict(lambda: 0)
        self._ep_ids = defaultdict(lambda: None)

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step)
        self.agent_step = 0
        self.exp.register_step("agent_step", lambda: self.agent_step)

    def _compile(self, model: nn.Module):
        return torch.compile(model, mode="reduce-overhead")

    def _make_agent(self, mode: Literal["train", "val"]):
        agent_ = rl_.Agent(
            self.actor,
            sample=(mode == "train"),
            compute_dtype=self.compute_dtype,
        )

        if self.cfg.rl.loader == "dream_rl":
            agent_ = wm.Agent(
                agent_,
                wm=self.wm,
                compute_dtype=self.compute_dtype,
            )

        noise = getattr(self.cfg, mode).agent_noise
        agent_ = agent.Agent(agent_, noise=noise, mode=mode)

        return agent_

    @argcast
    def _get_until_params(self, spec: config.Until):
        if isinstance(spec, int):
            max_value = spec
            of = self.cfg.def_step
        else:
            max_value, of = spec.n, spec.of
        return max_value, of

    @argcast
    def _make_until(self, spec: config.Until):
        max_value, of = self._get_until_params(spec)
        return cron.Until(lambda of=of: getattr(self, of), max_value)

    @argcast
    def _make_every(self, spec: config.Every):
        if isinstance(spec, int):
            spec = {"n": spec}
        else:
            spec = vars(spec)

        args = {**spec}
        args["period"] = args["n"]
        del args["n"]
        of = args["of"]
        if of is None:
            of = self.cfg.def_step
        del args["of"]
        args["step_fn"] = lambda of=of: getattr(self, of)

        return cron.Every(**args)

    @argcast
    def _make_sched(self, spec: config.Sched):
        if isinstance(spec, float):
            return sched.Constant(spec)
        else:
            if isinstance(spec, str):
                sched_fn, unit = spec, self.cfg.def_step
            else:
                sched_fn, unit = spec.value, spec.of
            return sched.Auto(sched_fn, lambda: getattr(self, unit))

    @exec_once
    def setup_train(self):
        cfg = self.cfg.data

        self.train_ids, self.val_ids = set(), set()
        self.train_ep_ids = rl.data.Sampler()
        self.val_ep_ids = rl.data.Sampler()

        class SplitHook(rl.data.Hook):
            def on_create(hook, seq_id: int):
                total = len(self.train_ids) + len(self.val_ids)
                is_val = len(self.val_ids) <= self.cfg.data.val_frac * total
                if is_val:
                    self.val_ep_ids.add(seq_id)
                    self.val_ids.add(seq_id)
                else:
                    self.train_ep_ids.add(seq_id)
                    self.train_ids.add(seq_id)

            def on_delete(hook, seq_id: int):
                if seq_id in self.val_ids:
                    del self.val_ep_ids[seq_id]
                    self.val_ids.remove(seq_id)
                else:
                    del self.train_ep_ids[seq_id]
                    self.train_ids.remove(seq_id)

        self.buf = rl.data.Observable(self.buf)
        self.buf.attach(SplitHook(), replay=True)

        if self.cfg.wm.loader == "real_wm":
            self.wm_loader = data.RealLoaderWM(
                buf=self.buf,
                sampler=self.train_ep_ids,
                **cfg.loaders.real_wm,
            )
            self.wm_iter = iter(self.wm_loader)

            if self.cfg.repro.determinism != "full":
                self.wm_iter = data.make_async(self.wm_iter)

        if self.cfg.rl.loader == "dream_rl":
            self.rl_loader = data.DreamLoaderRL(
                real_slices=self.wm_loader,
                wm=self.wm,
                actor=self.actor,
                device=self.device,
                compute_dtype=self.compute_dtype,
                **cfg.loaders.dream_rl,
            )
            self.rl_iter = iter(self.rl_loader)

        elif self.cfg.rl.loader == "real_rl":
            sig = inspect.signature(data.RealLoaderRL)
            args = sig.bind(
                buf=self.buf,
                sampler=self.train_ep_ids,
                **cfg.loaders.real_rl,
            )
            args.apply_defaults()

            sampler_type = args.arguments["sampler_type"]
            if sampler_type == "ep_ids":
                sampler = self.train_ep_ids
            else:
                sampler = rl.data.PSampler()

            self.rl_loader = data.RealLoaderRL(
                buf=self.buf,
                sampler=sampler,
                **cfg.loaders.real_rl,
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

            self.rl_loader = data.OnPolicyLoaderRL(
                do_env_step=self.do_env_step,
                temp_buf=temp_buf,
                **cfg.loaders.on_policy,
            )
            self.rl_iter = iter(self.rl_loader)

        self.buf = rl.data.SizeLimited(self.buf, cap=cfg.capacity)

        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer
            self.wm_trainer = dreamer.Trainer(wm_cfg, self.wm, self.compute_dtype)
        else:
            self.wm_trainer = None

        rl_type = self.cfg.rl.type
        if rl_type == "a2c":
            rl_cfg = self.cfg.rl.a2c
            self.rl_trainer = a2c.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
            )

        elif rl_type == "sac":
            rl_cfg = self.cfg.rl.sac
            self.rl_trainer = sac.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
            )

        elif rl_type == "ppo":
            rl_cfg = self.cfg.rl.ppo
            self.rl_trainer = ppo.Trainer(
                cfg=rl_cfg,
                actor=self.actor,
                compute_dtype=self.compute_dtype,
            )

        else:
            raise ValueError(rl_type)

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

    @exec_once
    def setup_val(self):
        if self.cfg.wm.loader == "real_wm":
            self.wm_val_step_loader = data.RealLoaderWM(
                buf=self.buf,
                sampler=self.val_ep_ids,
                **self.cfg.data.loaders.real_wm,
            )
            self.wm_val_iter = iter(self.wm_val_step_loader)

            if self.cfg.repro.determinism != "full":
                self.wm_val_iter = data.make_async(self.wm_val_iter)

            self.wm_val_epoch_loader = data.RealLoaderWM(
                buf=self.buf,
                sampler=self.val_ids,
                **self.cfg.data.loaders.real_wm,
            )

        if self.cfg.rl.loader == "dream_rl":
            self.rl_val_loader = data.DreamLoaderRL(
                real_slices=self.wm_val_step_loader,
                wm=self.wm,
                actor=self.actor,
                device=self.device,
                compute_dtype=self.compute_dtype,
                **self.cfg.data.loaders.dream_rl,
            )
            self.rl_val_iter = iter(self.rl_val_loader)

        self.val_envs = self.sdk.make_envs(self.cfg.val.num_envs, mode="val")
        self.val_agent = self._make_agent(mode="val")

    def save_buf(self, path: str | None = None, tag: str | None = None):
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
        state["actor"] = self.actor.state_dict()

        if full:
            if self.wm_trainer is not None:
                state["wm_trainer"] = self.wm_trainer.save()

            state = {
                **state,
                "rl_trainer": self.rl_trainer.save(),
                "repro": repro.state.save(),
            }

        if tag is None:
            cur_step = getattr(self, self.cfg.def_step)
            tag = f"ckpt.{self.cfg.def_step}={cur_step:g}"

        dst = self.exp.dir / "ckpts" / f"{tag}.pth"
        dst.parent.mkdir(parents=True, exist_ok=True)

        with open(dst, "wb") as f:
            torch.save(state, f)

        self.exp.log(logging.INFO, f"Saved checkpoint to: {str(dst)}")

    def load_ckpt(self, path: str | Path):
        path = Path(path)

        with open(path, "rb") as f:
            state = torch.load(f, map_location="cpu")

        if "wm" in state and self.wm is not None:
            self.wm.load_state_dict(state["wm"])

        self.actor.load_state_dict(state["actor"])

        if "wm_trainer" in state and self.wm_trainer is not None:
            self.wm_trainer.load(state["wm_trainer"])

        if "rl_trainer" in state:
            self.rl_trainer.load(state["rl_trainer"])

        if "repro" in state:
            repro.state.load(state["repro"])

        self.exp.log(logging.INFO, f"Loaded checkpoint from: {str(path)}")

    def env_rollout(self, until: config.Until, mode: str = "train"):
        should_do = self._make_until(until)

        pbar = self._make_pbar("Env rollout", until=until)
        while should_do:
            self.do_env_step(mode=mode)
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
        self.setup_train()
        self.setup_val()

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

        if self.cfg.wm.loader is not None:
            while self.wm_loader.empty():
                self.do_env_step()
                self._update_pbar(pbar, should_loop)

        if self.cfg.rl.loader is not None:
            while self.rl_loader.empty():
                self.do_env_step()
                self._update_pbar(pbar, should_loop)

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

    def do_opt_step(self, n=1):
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
        self.setup_val()
        _step = step

        if self.cfg.wm.loader is not None:
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

    def train_wm(
        self,
        stop_criteria: dict,
        val_every: int,
        val_on_loss_improv: float | None = None,
        max_val_batches: int | None = None,
    ):
        self.setup_train()

        should_stop = EarlyStopping(**stop_criteria)
        should_val = cron.Every(lambda: self.wm_opt_step, val_every)

        best_val_loss = np.inf
        best_ckpt = tempfile.mktemp(suffix=".pth")

        best_train_loss, prev_loss_val = None, None

        pbar = self._make_pbar("Train WM", of="wm_opt_step")
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
                        },
                        best_ckpt,
                    )

                if should_stop(val_loss, self.wm_opt_step):
                    break

            train_loss = self.do_wm_opt_step()
            if val_on_loss_improv is not None:
                if best_train_loss is None or train_loss < best_train_loss:
                    best_train_loss = train_loss

            pbar.update()

        with open(best_ckpt, "rb") as f:
            best_state = torch.load(f, map_location="cpu")
            self.wm.load_state_dict(best_state["wm"])
            self.wm_trainer.load(best_state["wm_trainer"])

        Path(best_ckpt).unlink()

    @torch.no_grad()
    def do_wm_val_epoch(self, max_batches=None):
        self.setup_val()

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

        val_loss = val_loss / val_n
        return val_loss

    def do_wm_opt_step(self, n=1):
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

    def do_rl_opt_step(self, n=1):
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

        if hasattr(self.wm_trainer, "compute_stats"):
            for k, v in self.wm_trainer.compute_stats().items():
                self.exp.add_scalar(f"wm/stats/{k}", v, step="wm_opt_step")

    def do_rl_val_step(self):
        if hasattr(self.rl_trainer, "compute_stats"):
            for k, v in self.rl_trainer.compute_stats().items():
                self.exp.add_scalar(f"rl/stats/{k}", v, step="rl_opt_step")

    def reset_rl(self):
        if self.cfg.rl.type == "sac":

            def make_actor():
                cfg = self.cfg.rl.sac
                actor = sac.Actor(cfg.actor, self.rl_obs_space, self.rl_act_space)
                actor = actor.to(self.device)
                return actor

            self.rl_trainer.reset(make_actor)
        else:
            raise ValueError(self.cfg.rl.type)
