import concurrent.futures
import logging
import lzma
import pickle
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from rsrch import spaces
from rsrch._future import rl
from rsrch._future.rl import buffer
from rsrch.exp import Experiment, timestamp
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.utils import cron, repro
from rsrch.utils.early_stop import EarlyStopping

from . import ac, agent, data, wm


@dataclass
class Config:
    @dataclass
    class Debug:
        deterministic: bool

    @dataclass
    class Profile:
        enabled: bool
        schedule: dict
        functions: list[str]

    @dataclass
    class Train:
        @dataclass
        class Dataset:
            capacity: int
            batch_size: int
            slice_len: int
            ongoing: bool = False
            subseq_len: int | tuple[int, int] | None = None
            prioritize_ends: bool = False

        dataset: Dataset
        val_frac: float
        num_envs: int

    seed: int
    device: str
    compute_dtype: Literal["float16", "float32"]
    def_step: str

    debug: Debug
    profile: Profile
    env: rl.api.Config
    wm: wm.Config
    ac: ac.Config
    agent: agent.Config
    train: Train

    stages: list[dict | str]


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
        repro.fix_seeds(
            seed=self.cfg.seed,
            deterministic=self.cfg.debug.deterministic,
        )

        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        self.env_api = rl.api.make(self.cfg.env)

        self.exp = Experiment(
            project="dreamerv2",
            run=f"{self.env_api.id}__{timestamp()}",
            config=asdict(self.cfg),
        )

        board = Tensorboard(dir=self.exp.dir / "board", min_delay=1e-1)
        self.exp.boards.append(board)
        self.exp.set_as_default(self.cfg.def_step)

        # We need the trainer to deduce the reward space, even if we don't
        # run the training.
        wm_trainer = wm.Trainer(self.cfg.wm, self.compute_dtype)
        rew_space = wm_trainer.reward_space

        obs_space, act_space = self.env_api.obs_space, self.env_api.act_space
        self.wm = wm.WorldModel(self.cfg.wm, obs_space, act_space, rew_space)
        self.wm = self.wm.to(self.device)
        self.wm.apply(self._tf_init)

        self.actor = ac.Actor(self.wm, self.cfg.ac)
        self.actor = self.actor.to(self.device)
        self.actor.apply(self._tf_init)

        self.fwd_mtx = Lock()

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

        self.buf = buffer.Buffer()
        self.buf_mtx = Lock()

        self.envs = self.env_api.make_envs(
            self.cfg.train.num_envs,
            mode="train",
        )
        self.agent = agent.Agent(
            actor=self.actor,
            wm=self.wm,
            cfg=self.cfg.agent,
            act_space=self.env_api.act_space,
            mode="train",
            compute_dtype=self.compute_dtype,
        )

        self.env_iter = iter(self.env_api.rollout(self.envs, self.agent))
        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)
        self._env_steps = defaultdict(lambda: 0)

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step)
        self.agent_step = 0
        self.exp.register_step("agent_step", lambda: self.agent_step)

    def _get_until_params(self, cfg: dict | str):
        if isinstance(cfg, dict):
            max_value, of = cfg["n"], cfg["of"]
        else:
            max_value = cfg
            of = self.cfg.def_step
        return max_value, of

    def _make_until(self, cfg):
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

    def _make_every(self, cfg):
        every, of, iters = self._get_every_params(cfg)
        return cron.Every(lambda of=of: getattr(self, of), every, iters)

    def _tf_init(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _setup_train(self):
        cfg = self.cfg.train

        dataset_cfg = {**vars(cfg.dataset)}
        self.buf = buffer.SizeLimited(self.buf, cap=dataset_cfg["capacity"])
        del dataset_cfg["capacity"]

        self.buf = self.env_api.wrap_buffer(self.buf)

        self.train_sampler = buffer.Sampler()
        self.val_ids = set()

        class SplitHook(buffer.Hook):
            def __init__(hook):
                hook._g = np.random.default_rng()

            def on_create(hook, seq_id: int, seq: dict):
                is_val = hook._g.random() < cfg.val_frac
                if is_val:
                    self.val_ids.add(seq_id)
                else:
                    self.train_sampler.add(seq_id)

            def on_delete(hook, seq_id: int, seq: dict):
                if seq_id in self.val_ids:
                    self.val_ids.remove(seq_id)
                else:
                    del self.train_sampler[seq_id]

        hook = SplitHook()
        for seq_id, seq in self.buf.items():
            hook.on_create(seq_id, seq)
        self.buf.hooks.append(hook)

        self.train_ds = data.Slices(
            buf=self.buf,
            sampler=self.train_sampler,
            **dataset_cfg,
        )

        self._train_batches = Queue(maxsize=1)

        def loader_fn():
            cfg = self.cfg.train
            batch_iter = iter(
                data.DataLoader(
                    dataset=self.train_ds,
                    batch_size=cfg.dataset.batch_size,
                    collate_fn=self.train_ds.collate_fn,
                )
            )

            while True:
                batch = next(batch_iter)
                self._train_batches.put(batch)

        self.loader_thread = Thread(target=loader_fn, daemon=True)
        self.loader_thread.start()

        def train_batches():
            while True:
                yield self._train_batches.get()

        self.train_loader = iter(train_batches())

        self.val_ds = data.Slices(
            buf=self.buf,
            sampler=self.val_ids,
            **dataset_cfg,
        )

        self.val_loader = data.DataLoader(
            dataset=self.val_ds,
            batch_size=cfg.dataset.batch_size,
            collate_fn=self.val_ds.collate_fn,
        )

        self.wm_trainer = wm.Trainer(self.cfg.wm, self.compute_dtype)
        self.wm_trainer.setup(self.wm)

        self.wm_opt_step = 0
        self.exp.register_step("wm_opt_step", lambda: self.wm_opt_step)

        self.ac_trainer = ac.Trainer(self.cfg.ac, self.compute_dtype)
        self.ac_trainer.setup(self.wm, self.actor, self._make_critic)

        self.ac_opt_step = 0
        self.exp.register_step("ac_opt_step", lambda: self.ac_opt_step)

    def _make_critic(self):
        critic = ac.Critic(self.wm, self.cfg.ac)
        critic = critic.to(self.device)
        critic.apply(self._tf_init)
        return critic

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
            "_full": full,
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

        if state["_full"]:
            self.wm_trainer.load(state["wm_trainer"])
            self.ac_trainer.load(state["ac_trainer"])
            self.buf.load(state["buf"])
            repro.state.load(state["repro"])

        self.exp.log(logging.INFO, f"Loaded checkpoint from: {str(path)}")

    def env_rollout(self, until, mode):
        should_do = self._make_until(until)
        self.agent.mode = mode

        pbar = self._make_pbar("Env rollout", until)
        while should_do:
            self.do_env_step()
            self._update_pbar(pbar, should_do)

    def _make_pbar(self, desc, until_cfg: dict | str):
        max_value, of = self._get_until_params(until_cfg)

        return self.exp.pbar(
            desc=desc,
            initial=getattr(self, of),
            total=max_value,
            unit=f" {of}",
        )

    def _update_pbar(self, pbar: tqdm, until: cron.Until):
        pbar.update(until.step_fn() - pbar.n)

    def do_env_step(self, mode="train"):
        with self.fwd_mtx:
            self.agent.mode = mode
            env_idx, (step, final) = next(self.env_iter)

        with self.buf_mtx:
            self._ep_ids[env_idx] = self.buf.push(self._ep_ids[env_idx], step, final)

        self._ep_rets[env_idx] += step["reward"]
        if final:
            self.exp.add_scalar("train/ep_ret", self._ep_rets[env_idx])
            del self._ep_ids[env_idx]
            del self._ep_rets[env_idx]

        self.agent_step += 1

        # Some environments (esp. Atari ones) skip frames (e.g. we do one agent
        # action and, say, four env steps follow.) For the purposes of benchmarking
        # we need to keep track of the "real" number of env steps.
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
        self._setup_train()

        task_functions = []
        should_run = []

        _, of = self._get_until_params(until)

        for task in tasks:
            if isinstance(task, str):
                task = {task: {}}
                every = 1
            else:
                task = {**task}
                every = task.get("every", 1)
                if "every" in task:
                    del task["every"]

            should_run_task = self._make_every({"n": every, "of": of})
            should_run.append(should_run_task)

            name = next(iter(task.keys()))
            params = task[name]
            if params is None:
                params = {}
            func = lambda name=name, params=params: getattr(self, name)(**params)
            task_functions.append(func)

        pool = ThreadPoolExecutor(len(tasks))
        futures = {}
        active_tasks = set()

        should_loop = self._make_until(until)

        pbar = self._make_pbar("Train loop", until)

        while should_loop:
            completed, _ = concurrent.futures.wait(
                futures,
                return_when="FIRST_COMPLETED",
            )
            for fut in completed:
                fut.result()
                task_idx = futures[fut]
                del futures[fut]
                active_tasks.remove(task_idx)

            for task_idx in range(len(tasks)):
                if task_idx in active_tasks:
                    continue

                if should_run[task_idx]:
                    fut = pool.submit(task_functions[task_idx])
                    futures[fut] = task_idx
                    active_tasks.add(task_idx)

            self._update_pbar(pbar, should_loop)

    def do_opt_step(self, n=1):
        if n > 1:
            for _ in range(n):
                self.do_opt_step()

        with self.buf_mtx:
            wm_batch = next(self.train_loader)

        wm_batch = self._move_to_device(wm_batch)

        with self.fwd_mtx:
            self.wm_trainer.train()
            wm_loss, wm_mets, states = self.wm_trainer.compute(wm_batch)

            self.ac_trainer.train()
            ac_loss, ac_mets = self.ac_trainer.compute(
                {
                    "initial": states.flatten(),
                    "term": wm_batch["term"].flatten(),
                }
            )

        self.wm_trainer.opt_step(wm_loss)
        self.ac_trainer.opt_step(ac_loss)

        self.train_ds.update_states(batch=wm_batch, final=states[-1])

        for k, v in wm_mets.items():
            self.exp.add_scalar(f"wm/{k}", v, step="wm_opt_step")
        for k, v in ac_mets.items():
            self.exp.add_scalar(f"ac/{k}", v, step="ac_opt_step")

        self.wm_opt_step += 1
        self.ac_opt_step += 1

    def val_epoch(self, limit=None, episodes=32):
        wm_loss = self.do_wm_val_epoch(limit)
        if wm_loss is not None:
            self.exp.add_scalar("wm/val_loss", wm_loss, step="wm_opt_step")

        ac_loss = self.do_ac_val_epoch(limit)
        if ac_loss is not None:
            self.exp.add_scalar("ac/val_loss", ac_loss, step="ac_opt_step")

        val_envs = self.env_api.make_envs(min(episodes, 8), mode="val")
        val_agent = agent.Agent(
            self.actor,
            self.wm,
            self.cfg.agent,
            self.env_api.act_space,
            mode="eval",
            compute_dtype=self.compute_dtype,
        )

        val_iter = iter(self.env_api.rollout(val_envs, val_agent))
        env_rets = defaultdict(lambda: 0.0)
        all_rets = []

        with self.fwd_mtx:
            for env_idx, (step, final) in val_iter:
                env_rets[env_idx] += step["reward"]
                if final:
                    all_rets.append(env_rets[env_idx])
                    del env_rets[env_idx]
                    if len(all_rets) >= 32:
                        break

        self.exp.add_scalar("val/mean_ep_ret", np.mean(all_rets))

    def train_wm(self, early_stop, val_every, val_limit=None):
        should_stop = EarlyStopping(**early_stop)
        should_val = cron.Every(lambda: self.wm_opt_step, val_every)

        best_loss, best_state = np.inf, None

        while True:
            if should_val:
                val_loss = self.do_wm_val_epoch(val_limit)
                self.exp.add_scalar("wm/val_loss", val_loss, step="wm_opt_step")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        "wm": self.wm.state_dict(),
                        "wm_trainer": self.wm_trainer.save(),
                    }

                if should_stop(val_loss, self.wm_opt_step):
                    break

            self.do_wm_opt_step()

        self.wm.load_state_dict(best_state["wm"])
        self.wm_trainer.load(best_state["wm_trainer"])

    @torch.no_grad()
    def do_wm_val_epoch(self, limit=None):
        val_loss, val_n = 0.0, 0

        val_iter = iter(islice(self.val_loader, 0, limit))

        while True:
            with self.buf_mtx:
                batch = next(val_iter, None)
                if batch is None:
                    break

            batch = self._move_to_device(batch)

            with self.fwd_mtx:
                self.wm_trainer.eval()
                loss, _, states = self.wm_trainer.compute(batch)

            with self.buf_mtx:
                self.val_ds.update_states(batch, states[-1])

            val_loss += loss
            val_n += 1

        if val_n > 0:
            val_loss = val_loss / val_n
        else:
            val_loss = None
        return val_loss

    def do_wm_opt_step(self, n=1):
        if n > 1:
            for _ in range(n):
                self.do_wm_opt_step()

        with self.buf_mtx:
            wm_batch = next(self.train_loader)

        wm_batch = self._move_to_device(wm_batch)
        with self.fwd_mtx:
            self.wm_trainer.train()
            loss, mets, states = self.wm_trainer.compute(wm_batch)

        self.wm_trainer.opt_step(loss)

        with self.buf_mtx:
            self.train_ds.update_states(batch=wm_batch, final=states[-1])

        for k, v in mets.items():
            self.exp.add_scalar(f"wm/{k}", v, step="wm_opt_step")

        self.wm_opt_step += 1

    def train_ac(self, early_stop, val_every, val_limit=None):
        should_stop = EarlyStopping(**early_stop)
        should_val = cron.Every(lambda: self.ac_opt_step, val_every)

        best_loss, best_state = np.inf, None

        while True:
            if should_val:
                val_loss = self.do_ac_val_epoch(val_limit)
                self.exp.add_scalar("ac/val_loss", val_loss, step="ac_opt_step")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        "actor": self.actor.state_dict(),
                        "ac_trainer": self.ac_trainer.save(),
                    }

                if should_stop(val_loss, self.wm_opt_step):
                    break

            self.do_ac_opt_step()

        self.actor.load_state_dict(best_state["actor"])
        self.ac_trainer.load(best_state["ac_trainer"])

    @torch.no_grad()
    def do_ac_val_epoch(self, limit=None):
        val_loss, val_n = 0.0, 0

        val_iter = iter(islice(self.val_loader, 0, limit))

        while True:
            with self.buf_mtx:
                wm_batch = next(val_iter, None)
                if wm_batch is None:
                    break

            wm_batch = self._move_to_device(wm_batch)

            with self.fwd_mtx:
                self.wm_trainer.eval()
                states = self.wm_trainer.get_states(wm_batch)

                self.ac_trainer.eval()
                loss, _ = self.ac_trainer.compute(
                    {
                        "initial": states.flatten(),
                        "term": wm_batch["term"].flatten(),
                    }
                )

            val_loss += loss
            val_n += 1

        if val_n > 0:
            val_loss = val_loss / val_n
        else:
            val_loss = None
        return val_loss

    def do_ac_opt_step(self, n=1):
        if n > 1:
            for _ in range(n):
                self.do_ac_opt_step()

        with self.buf_mtx:
            wm_batch = next(self.train_loader)

        wm_batch = self._move_to_device(wm_batch)

        with self.fwd_mtx:
            with torch.no_grad():
                self.wm_trainer.eval()
                states = self.wm_trainer.get_states(wm_batch)

            self.ac_trainer.train()
            loss, mets = self.ac_trainer.compute(
                {
                    "initial": states.flatten(),
                    "term": wm_batch["term"].flatten(),
                }
            )

        self.ac_trainer.opt_step(loss)

        for k, v in mets.items():
            self.exp.add_scalar(f"ac/{k}", v, step="ac_opt_step")

        self.ac_opt_step += 1

    def _move_to_device(self, batch: dict):
        res = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                v = v.to(self.device)
            res[k] = v
        return res
