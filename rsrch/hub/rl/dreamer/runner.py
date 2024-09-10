import concurrent.futures
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
from threading import Lock, Thread
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from rsrch import rl
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro
from rsrch.utils.early_stop import EarlyStopping

from . import agent, data
from . import wm as wm_
from .actor import ref
from .config import Config
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
                func = lambda: getattr(self, name)(**params)

            func()

    def _setup_common(self):
        repro.seed_all(self.cfg.repro.seed)
        repro.set_fully_deterministic(self.cfg.repro.determinism == "full")

        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.sdk = rl.sdk.make(self.cfg.env)

        if not self.sdk.act_space.dtype.is_floating_point:
            n = self.cfg.extras.discrete_actions
            if n is not None:
                self.sdk = rl.sdk.wrappers.DiscreteActions(self.sdk, n=n)

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

        self.wm = self._make_wm()
        self.actor = self._make_ac(self.wm)

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
        self.buf_mtx = Lock()

        self.envs = self.sdk.make_envs(
            self.cfg.train.num_envs,
            mode="train",
        )

        self.agent = self._make_agent(mode="train")

        self.env_iter = iter(self.sdk.rollout(self.envs, self.agent))
        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)
        self._env_steps = defaultdict(lambda: 0)

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step)
        self.agent_step = 0
        self.exp.register_step("agent_step", lambda: self.agent_step)

    def _make_wm(self):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer

            # We need the trainer to deduce the reward space, even if we don't
            # run the training.
            wm_trainer = dreamer.Trainer(wm_cfg, self.compute_dtype)
            rew_space = wm_trainer.reward_space

            obs_space, act_space = self.sdk.obs_space, self.sdk.act_space
            wm = dreamer.WorldModel(wm_cfg, obs_space, act_space, rew_space)
        else:
            raise ValueError(wm_type)

        wm = wm.to(self.device)
        return wm

    def _make_ac(self, wm: dreamer.WorldModel):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            Encoder = dreamer.AsTensor

        ac_type = self.cfg.ac.type
        if ac_type == "ref":
            ac_cfg = self.cfg.ac.ref
            actor = nn.Sequential(
                Encoder(),
                ref.Actor(ac_cfg, wm.state_size, wm.act_space),
            )
        else:
            raise ValueError(ac_type)

        actor = actor.to(self.device)
        return actor

    def _make_agent(self, mode: Literal["train", "val"]):
        agent_ = ref.Agent(
            self.actor,
            sample=(mode == "train"),
            compute_dtype=self.compute_dtype,
        )

        agent_ = wm_.Agent(
            agent_,
            wm=self.wm,
            compute_dtype=self.compute_dtype,
        )

        noise = getattr(self.cfg, mode).agent_noise
        agent_ = agent.Agent(
            agent_,
            act_space=self.sdk.act_space,
            noise=noise,
            mode=mode,
        )

        return agent_

    def _get_until_params(self, cfg: dict | str):
        if isinstance(cfg, dict):
            max_value, of = cfg["n"], cfg["of"]
        else:
            max_value = cfg
            of = self.cfg.def_step
        return max_value, of

    def _make_until(self, cfg: dict | str):
        max_value, of = self._get_until_params(cfg)
        return cron.Until(lambda of=of: getattr(self, of), max_value)

    def _get_every_params(self, cfg: dict | str):
        if isinstance(cfg, dict):
            every, of = cfg["n"], cfg["of"]
            iters = cfg.get("iters", 1)
        else:
            every = cfg
            of = self.cfg.def_step
            iters = 1
        return every, of, iters

    def _make_every(self, spec):
        if not isinstance(spec, dict):
            spec = {"n": spec}

        args = {**spec}
        args["period"] = args["n"]
        del args["n"]
        of = args.get("of", self.cfg.def_step)
        if "of" in args:
            del args["of"]
        args["step_fn"] = lambda of=of: getattr(self, of)
        if "iters" not in args:
            args["iters"] = 1

        return cron.Every(**args)

    @exec_once
    def setup_train(self):
        self.train_sampler = rl.data.Sampler()
        self.val_sampler = rl.data.Sampler()
        self.val_ids = set()

        class SplitHook(rl.data.Hook):
            def on_create(hook, seq_id: int):
                is_val = (len(self.val_ids) == 0) or (
                    np.random.rand() < self.cfg.data.val_frac
                )
                if is_val:
                    self.val_sampler.add(seq_id)
                    self.val_ids.add(seq_id)
                else:
                    self.train_sampler.add(seq_id)

            def on_delete(hook, seq_id: int):
                if seq_id in self.val_ids:
                    del self.val_sampler[seq_id]
                    self.val_ids.remove(seq_id)
                else:
                    del self.train_sampler[seq_id]

        self.buf = rl.data.Observable(self.buf)
        self.buf.attach(SplitHook(), replay=True)

        buf_cfg = {**vars(self.cfg.data.buffer)}
        self.buf = rl.data.SizeLimited(self.buf, cap=buf_cfg["capacity"])
        del buf_cfg["capacity"]
        self.buf_cfg = buf_cfg

        self.train_loader = data.RealLoaderWM(
            buf=self.buf,
            sampler=self.train_sampler,
            **self.buf_cfg,
        )
        self.train_iter = iter(self.train_loader)

        cfg = self.cfg.data.dream
        self.dream_loader = data.DreamLoaderRL(
            real_slices=self.train_loader,
            wm=self.wm,
            actor=self.actor,
            batch_size=cfg.batch_size,
            slice_len=cfg.horizon,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        self.dream_iter = iter(self.dream_loader)

        if self.cfg.repro.determinism != "full":
            self.train_iter = data.make_async(self.train_iter)

        self.wm_trainer = self._make_wm_trainer(self.wm)
        self.ac_trainer = self._make_ac_trainer(self.actor)

        self.wm_opt_step = 0
        self.exp.register_step("wm_opt_step", lambda: self.wm_opt_step)

        self.ac_opt_step = 0
        self.exp.register_step("ac_opt_step", lambda: self.ac_opt_step)

        self.should_log_wm = cron.Every(
            lambda: self.wm_opt_step,
            period=self.cfg.run.log_every,
            iters=None,
        )
        self.should_log_ac = cron.Every(
            lambda: self.ac_opt_step,
            period=self.cfg.run.log_every,
            iters=None,
        )

    def _make_wm_trainer(self, wm):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm_cfg = self.cfg.wm.dreamer
            wm_trainer = dreamer.Trainer(wm_cfg, self.compute_dtype)
            wm_trainer.setup(wm)
        else:
            raise ValueError(wm_type)

        return wm_trainer

    def _make_ac_trainer(self, actor):
        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            Encoder = dreamer.AsTensor

        ac_type = self.cfg.ac.type
        if ac_type == "ref":
            ac_cfg = self.cfg.ac.ref
            ac_trainer = ref.Trainer(ac_cfg, self.compute_dtype)

            def make_critic():
                critic = nn.Sequential(
                    Encoder(),
                    ref.Critic(ac_cfg, self.wm.state_size),
                )
                critic = critic.to(self.device)
                return critic

            ac_trainer.setup(actor, make_critic, self.sdk.act_space)

        return ac_trainer

    @exec_once
    def setup_val(self):
        self.val_loader = data.RealLoaderWM(
            buf=self.buf,
            sampler=self.val_sampler,
            **self.buf_cfg,
        )
        self.val_iter = iter(self.val_loader)

        cfg = self.cfg.data.dream
        self.val_dream_loader = data.DreamLoaderRL(
            real_slices=self.val_loader,
            wm=self.wm,
            actor=self.actor,
            batch_size=cfg.batch_size,
            slice_len=cfg.horizon,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )
        self.val_dream_iter = iter(self.val_dream_loader)

        self.val_epoch_loader = data.RealLoaderWM(
            buf=self.buf,
            sampler=self.val_ids,
            **self.buf_cfg,
        )

        if self.cfg.repro.determinism != "full":
            self.val_iter = data.make_async(self.val_iter)

        self.val_envs = self.sdk.make_envs(
            self.cfg.val.num_envs,
            mode="val",
        )

        self.val_agent = self._make_agent(mode="val")

    def save_data(self, tag: str | None = None):
        if tag is None:
            cur_step = getattr(self, self.cfg.def_step)
            tag = f"{self.cfg.def_step}={cur_step:g}"

        dst = self.exp.dir / "data" / f"samples.{tag}.pkl.lzma"
        dst.parent.mkdir(parents=True, exist_ok=True)

        with lzma.open(dst, "wb") as f:
            pickle.dump(self.buf.save(), f)

        self.exp.log(logging.INFO, f"Saved buffer data to: {str(dst)}")

    def load_data(self, path: str | Path):
        path = Path(path)

        with lzma.open(path, "rb") as f:
            self.buf.load(pickle.load(f))

        self.exp.log(logging.INFO, f"Loaded buffer data from: {str(path)}")

    def save_ckpt(self, full: bool = False):
        state = {
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
        }
        if full:
            state = {
                **state,
                "wm_trainer": self.wm_trainer.save(),
                "ac_trainer": self.ac_trainer.save(),
                "buf": self.buf.save(),
                "repro": repro.state.save(),
            }

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

        self.wm.load_state_dict(state["wm"])
        self.actor.load_state_dict(state["actor"])

        if "buf" in state:
            self.wm_trainer.load(state["wm_trainer"])
            self.ac_trainer.load(state["ac_trainer"])
            self.buf.load(state["buf"])
            repro.state.load(state["repro"])

        self.exp.log(logging.INFO, f"Loaded checkpoint from: {str(path)}")

    def env_rollout(self, until, mode: str = "train"):
        should_do = self._make_until(until)

        pbar = self._make_pbar("Env rollout", until=until)
        while should_do:
            self.do_env_step(mode=mode)
            self._update_pbar(pbar, should_do)

    def _make_pbar(
        self,
        desc: str,
        *,
        until: dict | str | None = None,
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

        self._ep_rets[env_idx] += step.get("reward", 0.0)
        if final:
            self.exp.add_scalar("train/ep_ret", self._ep_rets[env_idx])
            del self._ep_ids[env_idx]
            del self._ep_rets[env_idx]

        self.agent_step += 1

        # Some environments skip frames (each action is repeated, say, four times.) For the purposes of benchmarking and keeping the comparisons fair, we need to keep track of the "real" number of env steps. This is done by adding "total_steps" field for all steps produced by the SDKs.
        if "total_steps" in step:
            diff = step["total_steps"] - self._env_steps[env_idx]
            self.env_step += diff
            self._env_steps[env_idx] = step["total_steps"]
        else:
            self.env_step += 1

    def train_loop(
        self,
        until: dict | str,
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
            func = lambda name=name, params=params: getattr(self, name)(**params)
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

    def do_opt_step(self, n=1):
        for _ in range(n):
            self.do_wm_opt_step()
            self.do_ac_opt_step()

    def do_val_step(self):
        if len(self.val_ids) == 0:
            return

        self.wm_trainer.eval()
        self.ac_trainer.eval()

        with self.buf_mtx:
            wm_batch = next(self.val_iter)

        wm_batch = wm_batch.to(self.device)

        wm_output = self.wm_trainer.compute(
            seq=wm_batch.seq,
            h_0=wm_batch.h_0,
        )

        ac_batch = self.dream_loader.dream_from(
            wm_output.states.flatten(),
            wm_batch.seq.term.flatten(),
        )

        ac_output = self.ac_trainer.compute(ac_batch)

        with self.buf_mtx:
            self.val_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

        self.exp.add_scalar(f"wm/val_loss", wm_output.loss, step="wm_opt_step")
        self.exp.add_scalar(f"ac/val_loss", ac_output.loss, step="ac_opt_step")

    def do_val_epoch(self, max_batches=None):
        self.setup_val()

        wm_loss = self.do_wm_val_epoch(max_batches)
        if wm_loss is not None:
            self.exp.add_scalar("val/wm_loss", wm_loss, step="wm_opt_step")

        val_iter = iter(self.sdk.rollout(self.val_envs, self.val_agent))
        env_rets = defaultdict(lambda: 0.0)
        all_rets = []

        for env_idx, (step, final) in val_iter:
            env_rets[env_idx] += step["reward"]
            if final:
                all_rets.append(env_rets[env_idx])
                del env_rets[env_idx]
                if len(all_rets) >= 32:
                    break

        self.exp.add_scalar("val/mean_ep_ret", np.mean(all_rets))

    def train_wm(
        self,
        stop_criteria: dict,
        val_every: int,
        val_on_loss_improv: float | None = None,
        max_val_batches: int | None = None,
        reset: bool = True,
    ):
        self.setup_train()

        should_stop = EarlyStopping(**stop_criteria)
        should_val = cron.Every(lambda: self.wm_opt_step, val_every)

        best_val_loss = np.inf
        best_ckpt = tempfile.mktemp(suffix=".pth")

        best_train_loss, prev_loss_val = None, None

        if reset:
            self.reset_wm()

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

        self.val_epoch_loader.h_0s.clear()
        val_iter = iter(islice(self.val_epoch_loader, 0, max_batches))

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
                self.val_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

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
            wm_batch = next(self.train_iter)

        wm_batch = wm_batch.to(self.device)

        wm_output = self.wm_trainer.compute(
            seq=wm_batch.seq,
            h_0=wm_batch.h_0,
        )

        self.wm_trainer.opt_step(wm_output.loss)

        self.dream_loader.to_recycle = (
            wm_output.states.flatten(),
            wm_batch.seq.term.flatten(),
        )

        with self.buf_mtx:
            self.train_loader.h_0s[wm_batch.end_pos] = wm_output.h_n

        if self.should_log_wm:
            for k, v in wm_output.metrics.items():
                self.exp.add_scalar(f"wm/{k}", v, step="wm_opt_step")
            self.exp.add_scalar(f"wm/env_step", self.env_step, step="wm_opt_step")

        self.wm_opt_step += 1
        return wm_output.loss

    def do_ac_opt_step(self, n=1):
        if n > 1:
            for _ in range(n):
                self.do_ac_opt_step()

        with self.buf_mtx:
            wm_batch = next(self.train_iter)

        wm_batch = wm_batch.to(self.device)

        self.ac_trainer.train()

        ac_batch = next(self.dream_iter)

        ac_output = self.ac_trainer.compute(ac_batch)

        self.ac_trainer.opt_step(ac_output.loss)

        if self.should_log_ac:
            for k, v in ac_output.metrics.items():
                self.exp.add_scalar(f"ac/{k}", v, step="ac_opt_step")
            self.exp.add_scalar(f"ac/env_step", self.env_step, step="ac_opt_step")

        self.ac_opt_step += 1

    def reset_wm(self):
        # We create a new WM and copy the parameters, so that the
        # references in other objects (e.g. the agent) stay the same.
        new_wm = self._make_wm()
        polyak.sync(new_wm, self.wm)
        self.wm_trainer.setup(self.wm)
