import concurrent.futures
import logging
import lzma
import pickle
from collections import defaultdict
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

from rsrch._future import rl
from rsrch.exp import Experiment, timestamp, timestamp2
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.nn.utils import over_seq
from rsrch.types.tensorlike.core import Tensorlike
from rsrch.utils import cron, repro
from rsrch.utils.early_stop import EarlyStopping

from . import agent, data
from .actor import ref, sac
from .common.utils import autocast
from .wm import dreamer


@dataclass
class Config:
    @dataclass
    class Debug:
        deterministic: bool
        detect_anomaly: bool

    @dataclass
    class Profile:
        enabled: bool
        schedule: dict
        functions: list[str]

    @dataclass
    class WM:
        type: Literal["dreamer"]
        dreamer: dreamer.Config

    @dataclass
    class Actor:
        type: Literal["ref", "sac"]
        ref: ref.Config | None
        sac: sac.Config | None

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
        horizon: int

    @dataclass
    class Extras:
        discrete_actions: int | None = None

    seed: int
    device: str
    compute_dtype: Literal["float16", "float32"]
    def_step: str
    val_envs: int
    extras: Extras

    debug: Debug
    profile: Profile
    env: rl.sdk.Config
    wm: WM
    ac: Actor
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

        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        self.sdk = rl.sdk.make(self.cfg.env)
        if not self.sdk.act_space.dtype.is_floating_point:
            n = self.cfg.extras.discrete_actions
            if n is not None:
                self.sdk = rl.sdk.wrappers.DiscreteActions(self.sdk, n=n)

        self.exp = Experiment(
            project="dreamerv2",
            prefix=self.sdk.id,
            config=asdict(self.cfg),
        )

        board = Tensorboard(dir=self.exp.dir / "board", min_delay=1e-1)
        self.exp.boards.append(board)
        self.exp.set_as_default(self.cfg.def_step)

        if self.cfg.wm.type == "dreamer":
            wm = dreamer
            wm_cfg = self.cfg.wm.dreamer

            # We need the trainer to deduce the reward space, even if we don't
            # run the training.
            wm_trainer = wm.Trainer(wm_cfg, self.compute_dtype)
            rew_space = wm_trainer.reward_space

            obs_space, act_space = self.sdk.obs_space, self.sdk.act_space
            self.wm = wm.WorldModel(wm_cfg, obs_space, act_space, rew_space)
            self.wm = self.wm.to(self.device)

        ac_type = self.cfg.ac.type
        if ac_type == "ref":
            ac_cfg = self.cfg.ac.ref
            self.actor = nn.Sequential(
                wm.AsTensor(),
                ref.Actor(ac_cfg, self.wm.state_size, self.wm.act_space),
            )
        elif ac_type == "sac":
            ac_cfg = self.cfg.ac.sac
            self.actor = nn.Sequential(
                wm.AsTensor(),
                sac.Actor(ac_cfg, self.wm.state_size, self.wm.act_space),
            )

        self.actor = self.actor.to(self.device)

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

        self.buf = rl.data.Buffer()
        self.buf_mtx = Lock()

        self.envs = self.sdk.make_envs(
            self.cfg.train.num_envs,
            mode="train",
        )
        self.agent = agent.Agent(
            actor=self.actor,
            wm=self.wm,
            cfg=self.cfg.agent,
            act_space=self.sdk.act_space,
            mode="train",
            compute_dtype=self.compute_dtype,
        )

        self.env_iter = iter(self.sdk.rollout(self.envs, self.agent))
        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)
        self._env_steps = defaultdict(lambda: 0)

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step)
        self.agent_step = 0
        self.exp.register_step("agent_step", lambda: self.agent_step)

        self.val_envs = self.sdk.make_envs(self.cfg.val_envs, mode="val")
        self.val_agent = agent.Agent(
            self.actor,
            self.wm,
            self.cfg.agent,
            self.sdk.act_space,
            mode="eval",
            compute_dtype=self.compute_dtype,
        )

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

    def _setup_train(self):
        cfg = self.cfg.train

        dataset_cfg = {**vars(cfg.dataset)}
        self.buf = rl.data.SizeLimited(self.buf, cap=dataset_cfg["capacity"])
        del dataset_cfg["capacity"]

        self.buf = self.sdk.wrap_buffer(self.buf)

        self.train_sampler = rl.data.Sampler()
        self.val_ids = set()

        class SplitHook(rl.data.Hook):
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
            batch_size=cfg.dataset.batch_size,
            slice_len=cfg.dataset.slice_len,
        )

        self.val_loader = data.DataLoader(
            dataset=self.val_ds,
            batch_size=cfg.dataset.batch_size,
            collate_fn=self.val_ds.collate_fn,
            drop_last=False,
        )

        wm_type = self.cfg.wm.type
        if wm_type == "dreamer":
            wm = dreamer
            wm_cfg = self.cfg.wm.dreamer
            self.wm_trainer = dreamer.Trainer(wm_cfg, self.compute_dtype)
            self.wm_trainer.setup(self.wm)

        self.wm_opt_step = 0
        self.exp.register_step("wm_opt_step", lambda: self.wm_opt_step)

        ac_type = self.cfg.ac.type
        if ac_type == "ref":
            ac_cfg = self.cfg.ac.ref
            self.ac_trainer = ref.Trainer(ac_cfg, self.compute_dtype)

            def make_critic():
                critic = nn.Sequential(
                    wm.AsTensor(),
                    ref.Critic(ac_cfg, self.wm.state_size),
                )
                critic = critic.to(self.device)
                return critic

            self.ac_trainer.setup(self.actor, make_critic, self.sdk.act_space)

        elif ac_type == "sac":
            ac_cfg = self.cfg.ac.sac
            self.ac_trainer = sac.Trainer(ac_cfg, self.compute_dtype)

            def make_vf():
                vf = nn.Sequential(
                    wm.AsTensor(),
                    sac.ValueFunc(ac_cfg, self.wm.state_size),
                )
                vf = vf.to(self.device)
                return vf

            self.ac_trainer.setup(self.actor, make_vf)

        self.ac_opt_step = 0
        self.exp.register_step("ac_opt_step", lambda: self.ac_opt_step)

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

    def env_rollout(self, until, mode="train"):
        should_do = self._make_until(until)

        pbar = self._make_pbar("Env rollout", until)
        while should_do:
            self.do_env_step(mode=mode)
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
        if not hasattr(self, "_setup_train_done"):
            self._setup_train()
            self._setup_train_done = True

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

            if not isinstance(every, dict):
                # If only a number is supplied, use default 'of' value.
                every = dict(n=every, of=of)

            should_run_task = self._make_every(every)
            should_run.append(should_run_task)

            name = next(iter(task.keys()))
            params = task[name]
            if params is None:
                params = {}
            func = lambda name=name, params=params: getattr(self, name)(**params)
            task_functions.append(func)

        should_loop = self._make_until(until)
        pbar = self._make_pbar("Train loop", until)

        async_mode = not self.cfg.profile.enabled

        if async_mode:
            pool = concurrent.futures.ThreadPoolExecutor(len(tasks))
            fut_to_idx, idx_to_fut = {}, {}

            while should_loop:
                completed, _ = concurrent.futures.wait(
                    fut_to_idx,
                    return_when="FIRST_COMPLETED",
                )
                for fut in completed:
                    fut.result()
                    task_idx = fut_to_idx[fut]
                    del fut_to_idx[fut]
                    del idx_to_fut[task_idx]

                for task_idx in range(len(tasks)):
                    if should_run[task_idx]:
                        if task_idx in idx_to_fut:
                            fut = idx_to_fut[task_idx]
                            fut.result()
                            del fut_to_idx[fut]
                            del idx_to_fut[task_idx]

                        fut = pool.submit(task_functions[task_idx])
                        fut_to_idx[fut] = task_idx
                        idx_to_fut[task_idx] = fut

                self._update_pbar(pbar, should_loop)

            concurrent.futures.wait(fut_to_idx)

        else:
            while should_loop:
                for task_idx in range(len(tasks)):
                    if should_run[task_idx]:
                        task_functions[task_idx]()
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
            self.ac_trainer.train()

            wm_loss, wm_mets, wm_states = self.wm_trainer.compute(
                obs=wm_batch["obs"],
                act=wm_batch["act"],
                reward=wm_batch["reward"],
                term=wm_batch["term"],
                start=wm_batch["start"],
            )

            (ac_states, action, reward, term) = self._get_dream_batch(
                initial=wm_states.flatten(),
                term=wm_batch["term"].flatten(),
            )

            if self.cfg.ac.type == "sac":
                reward = reward[1:]

            ac_loss, ac_mets = self.ac_trainer.compute(
                states=ac_states,
                action=action,
                reward=reward,
                term=term,
            )

        self.ac_trainer.opt_step(ac_loss, self.fwd_mtx)
        self.wm_trainer.opt_step(wm_loss, self.fwd_mtx)

        self.train_ds.update_states(batch=wm_batch, final=wm_states[-1])

        for k, v in wm_mets.items():
            self.exp.add_scalar(f"wm/{k}", v, step="wm_opt_step")
        for k, v in ac_mets.items():
            self.exp.add_scalar(f"ac/{k}", v, step="ac_opt_step")

        self.wm_opt_step += 1
        self.ac_opt_step += 1

    def _get_dream_batch(self, initial, term):
        self.wm.requires_grad_(False)
        with autocast(self.device, self.compute_dtype):
            states, actions = [initial], []
            for _ in range(self.cfg.train.horizon):
                policy = self.actor(states[-1].detach())
                enc_act = policy.rsample()
                actions.append(enc_act)
                next_state = self.wm.img_step(states[-1], enc_act)
                states.append(next_state)
            states, actions = torch.stack(states), torch.stack(actions)

            reward_dist = over_seq(self.wm.reward_dec)(states)
            reward = reward_dist.mode

            term_dist = over_seq(self.wm.term_dec)(states)
            term_ = term_dist.mean.contiguous()
            term_[0] = term.float()
        self.wm.requires_grad_(True)

        return states, actions, reward, term_

    def val_epoch(self, limit=None):
        wm_loss = self.do_wm_val_epoch(limit)
        if wm_loss is not None:
            self.exp.add_scalar("wm/val_loss", wm_loss, step="wm_opt_step")

        ac_loss = self.do_ac_val_epoch(limit)
        if ac_loss is not None:
            self.exp.add_scalar("ac/val_loss", ac_loss, step="ac_opt_step")

        val_iter = iter(self.sdk.rollout(self.val_envs, self.val_agent))
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

        self.val_ds.states.clear()
        val_iter = iter(islice(self.val_loader, 0, limit))

        while True:
            with self.buf_mtx:
                wm_batch = next(val_iter, None)
                if wm_batch is None:
                    break

            wm_batch = self._move_to_device(wm_batch)

            with self.fwd_mtx:
                self.wm_trainer.eval()
                wm_loss, wm_mets, wm_states = self.wm_trainer.compute(
                    obs=wm_batch["obs"],
                    act=wm_batch["act"],
                    reward=wm_batch["reward"],
                    term=wm_batch["term"],
                    start=wm_batch["start"],
                )

            with self.buf_mtx:
                self.val_ds.update_states(wm_batch, wm_states[-1])

            val_loss += wm_loss
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
            wm_loss, wm_mets, wm_states = self.wm_trainer.compute(
                obs=wm_batch["obs"],
                act=wm_batch["act"],
                reward=wm_batch["reward"],
                term=wm_batch["term"],
                start=wm_batch["start"],
            )

        self.wm_trainer.opt_step(wm_loss, self.fwd_mtx)

        with self.buf_mtx:
            self.train_ds.update_states(batch=wm_batch, final=wm_states[-1])

        for k, v in wm_mets.items():
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

        self.val_ds.states.clear()
        val_iter = iter(islice(self.val_loader, 0, limit))

        while True:
            with self.buf_mtx:
                wm_batch = next(val_iter, None)
                if wm_batch is None:
                    break

            wm_batch = self._move_to_device(wm_batch)

            with self.fwd_mtx:
                self.wm.eval()
                self.ac_trainer.eval()

                wm_states = self.wm.observe(
                    obs=wm_batch["obs"],
                    act=wm_batch["act"],
                    start=wm_batch["start"],
                )[0]

                (ac_states, action, reward, term) = self._get_dream_batch(
                    initial=wm_states.flatten(),
                    term=wm_batch["term"].flatten(),
                )

                if self.cfg.ac.type == "sac":
                    reward = reward[1:]

                loss, _ = self.ac_trainer.compute(
                    states=ac_states,
                    action=action,
                    reward=reward,
                    term=term,
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
            self.wm.eval()
            self.ac_trainer.train()

            with torch.no_grad():
                wm_states = self.wm.observe(
                    obs=wm_batch["obs"],
                    act=wm_batch["act"],
                    start=wm_batch["start"],
                )[0]

            (ac_states, action, reward, term) = self._get_dream_batch(
                initial=wm_states.flatten(),
                term=wm_batch["term"].flatten(),
            )

            if self.cfg.ac.type == "sac":
                reward = reward[1:]

            ac_loss, ac_mets = self.ac_trainer.compute(
                states=ac_states,
                action=action,
                reward=reward,
                term=term,
            )

        self.ac_trainer.opt_step(ac_loss, self.fwd_mtx)

        for k, v in ac_mets.items():
            self.exp.add_scalar(f"ac/{k}", v, step="ac_opt_step")

        self.ac_opt_step += 1

    def _move_to_device(self, batch: dict):
        res = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                v = v.to(device=self.device)
            res[k] = v
        return res
