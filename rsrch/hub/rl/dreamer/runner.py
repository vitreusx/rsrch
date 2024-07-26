from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch._future import rl
from rsrch._future.rl import buffer
from rsrch.exp import Experiment, timestamp
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.utils import cron, repro

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
        agent: agent.Config
        num_envs: int
        prefill: int | dict
        total: int | dict
        opt_every: int | dict
        log_every: int

    seed: int
    device: str
    compute_dtype: Literal["float16", "float32"]

    debug: Debug
    profile: Profile
    env: rl.api.Config
    wm: wm.Config
    ac: ac.Config

    mode: Literal["train"]
    train: Train


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def main(self):
        self._setup_common()
        if self.cfg.mode == "train":
            self._setup_train()
            self.train()

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
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))

        self.pbar = self.exp.pbar(desc="DreamerV2")

        # We need the trainer to deduce the reward space, even if we don't
        # run the training.
        self.wm_trainer = wm.Trainer(self.cfg.wm, self.compute_dtype)

        obs_space, act_space = self.env_api.obs_space, self.env_api.act_space
        rew_space = self.wm_trainer.reward_space
        self.wm = wm.WorldModel(self.cfg.wm, obs_space, act_space, rew_space)
        self.wm = self.wm.to(self.device)
        self.wm.apply(self._tf_init)

        self.actor = ac.Actor(self.wm, self.cfg.ac)
        self.actor = self.actor.to(self.device)
        self.actor.apply(self._tf_init)

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

    def _tf_init(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def save(self, full=False):
        state = {"wm": self.wm.state_dict(), "actor": self.actor.state_dict()}
        if full:
            state += {
                "wm_trainer": self.wm_trainer.save(),
                "ac_trainer": self.ac_trainer.save(),
                "data": self.buf.save(),
                "repro": repro.state.save(),
            }

    def load(self, ckpt, full=False):
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        if full:
            self.wm_trainer.load(ckpt["wm_trainer"])
            self.ac_trainer.load(ckpt["ac_trainer"])
            self.buf.load(ckpt["data"])
            repro.state.load(ckpt["repro"])

    def _setup_train(self):
        cfg = self.cfg.train

        self.buf = buffer.Buffer()

        dataset_cfg = {**vars(cfg.dataset)}
        self.buf = buffer.SizeLimited(self.buf, cap=dataset_cfg["capacity"])
        del dataset_cfg["capacity"]

        self.buf = self.env_api.wrap_buffer(self.buf)

        self.train_sampler = buffer.Sampler()
        val_ids = set()

        class SplitHook(buffer.Hook):
            def __init__(hook):
                hook._g = np.random.default_rng()

            def on_create(hook, seq_id: int, seq: dict):
                is_val = hook._g.random() < cfg.val_frac
                if is_val:
                    val_ids.add(seq_id)
                else:
                    self.train_sampler.add(seq_id)

            def on_delete(hook, seq_id: int, seq: dict):
                if seq_id in val_ids:
                    val_ids.remove(seq_id)
                else:
                    del self.train_sampler[seq_id]

        self.buf.hooks.append(SplitHook())

        self.train_ds = data.Slices(
            buf=self.buf,
            sampler=self.train_sampler,
            **dataset_cfg,
        )

        self._train_batches = Queue(maxsize=1)
        self.train_loader = Thread(target=self._train_loader_fn)

        def train_batches():
            while True:
                yield self._train_batches.get()

        self.batch_iter = iter(train_batches())

        self.wm_trainer.setup(self.wm)

        self.ac_trainer = ac.Trainer(self.cfg.ac, self.compute_dtype)
        self.ac_trainer.setup(self.wm, self.actor, self._make_critic)

        self.agent = agent.Agent(
            actor=self.actor,
            cfg=cfg.agent,
            act_space=self.env_api.act_space,
        )
        self.envs = self.env_api.make_envs(cfg.num_envs, mode="train")
        self.env_iter = iter(self.env_api.rollout(self.envs, self.agent))

        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)
        self._env_steps = defaultdict(lambda: 0)

        self._prev_states = {}

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.agent_step = 0

        self.buf_mtx = Lock()
        self.fwd_mtx = Lock()
        self.log_mtx = Lock()

        self.log_every = cfg.log_every

    def _train_loader_fn(self):
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

    def _make_critic(self):
        critic = ac.Critic(self.wm, self.cfg.ac)
        critic = critic.to(self.device)
        critic.apply(self._tf_init)
        return critic

    def train(self):
        cfg = self.cfg.train

        self.agent.mode = "prefill"
        should_prefill = self._make_until(cfg.prefill)

        while should_prefill:
            self.take_env_step()

        self.agent.mode = "expl"
        should_train = self._make_until(cfg.total)

        self.opt_step = 0
        should_opt = self._make_every(cfg.opt_every)

        self.train_loader.start()

        def env_task():
            while True:
                self.take_env_step()
                if should_opt:
                    break

        pool = ThreadPoolExecutor(1)

        while should_train:
            fut = pool.submit(env_task)
            self._opt_step()
            fut.result()

    def _opt_step(self):
        with self.buf_mtx:
            wm_batch = next(self.batch_iter)

        wm_batch = self._move_to_device(wm_batch)

        for pos in [*self._prev_states]:
            if pos not in wm_batch["pos"]:
                del self._prev_states[pos]

        starts = []
        for pos in wm_batch["pos"]:
            state = self._prev_states.get(pos, None)
            starts.append(state)
        wm_batch["start"] = starts

        with self.fwd_mtx:
            loss, metrics, states = self.wm_trainer.compute(wm_batch, train=True)

        self.wm_trainer.opt_step(loss)

        slice_len = len(states)
        for idx, (seq_id, offset) in enumerate(wm_batch["pos"]):
            last = offset + slice_len
            self._prev_states[seq_id, last] = states[-1, idx]

        if self.opt_step % self.log_every == 0:
            with self.log_mtx:
                for k, v in metrics.items():
                    self.exp.add_scalar(f"wm/{k}", v)

        ac_batch = {
            "initial": states.flatten(),
            "term": wm_batch["term"].flatten(),
        }

        with self.fwd_mtx:
            loss, metrics = self.ac_trainer.compute(ac_batch, train=True)

        self.ac_trainer.opt_step(loss)

        if self.opt_step % self.log_every == 0:
            with self.log_mtx:
                for k, v in metrics.items():
                    self.exp.add_scalar(f"ac/{k}", v)

        self.opt_step += 1

    def _move_to_device(self, batch: dict):
        res = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                v = v.to(self.device)
            res[k] = v
        return res

    def _make_until(self, cfg):
        if isinstance(cfg, dict):
            n = cfg["n"]
            step_fn = lambda: getattr(self, cfg["of"])
        else:
            n = cfg
            step_fn = lambda: self.env_step
        return cron.Until(step_fn, n)

    def _make_every(self, cfg):
        if isinstance(cfg, dict):
            every = cfg["n"]
            step_fn = lambda: getattr(self, cfg["of"])
            iters = cfg.get("iters", 1)
        else:
            every = cfg
            step_fn = lambda: self.env_step
            iters = 1
        return cron.Every(step_fn, every, iters)

    def take_env_step(self):
        with self.fwd_mtx:
            env_idx, (step, final) = next(self.env_iter)

        with self.buf_mtx:
            self._ep_ids[env_idx] = self.buf.push(self._ep_ids[env_idx], step, final)

        self._ep_rets[env_idx] += step["reward"]
        if final:
            with self.log_mtx:
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
            diff = 1
            self.env_step += 1

        self.pbar.update(diff)
