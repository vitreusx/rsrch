from collections import defaultdict
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch._future import rl
from rsrch._future.rl import buffer
from rsrch.exp import Experiment, timestamp
from rsrch.utils import cron, repro

from . import ac, agent, data, wm


@dataclass
class Config:
    @dataclass
    class Debug:
        deterministic: bool

    @dataclass
    class Train:
        buffer: dict
        val_frac: float
        agent: agent.Config
        num_envs: int
        prefill: dict
        total: dict

    seed: int
    device: str
    compute_dtype: Literal["float16", "float32"]

    debug: Debug
    env: rl.api.Config
    wm: wm.Config
    ac: ac.Config

    mode: Literal["train"]
    train: Train


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def main(self):
        self._setup_core()
        if self.cfg.mode == "train":
            self._setup_train()
            self.train()

    def _setup_core(self):
        repro.fix_seeds(
            seed=self.cfg.seed,
            deterministic=self.cfg.debug.deterministic,
        )

        self.device = torch.device(self.cfg.device)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        self.exp = Experiment(
            project="dreamerv2",
            run=f"{self.cfg.env.id}__{timestamp()}",
            config=asdict(self.cfg),
        )

        self.pbar = self.exp.pbar(desc="DreamerV2")

        self.env_api = rl.api.make(self.cfg.env)

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
        self.wm.apply(self._tf_init)

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
                "repro": repro.RandomState().save(),
            }

    def load(self, ckpt, full=False):
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        if full:
            self.wm_trainer.load(ckpt["wm_trainer"])
            self.ac_trainer.load(ckpt["ac_trainer"])
            self.buf.load(ckpt["data"])
            repro.RandomState().load(ckpt["repro"])

    def _setup_train(self):
        cfg = self.cfg.train

        self.buf = buffer.Buffer()

        buf_cfg = {**cfg.buffer}
        self.buf = buffer.SizeLimited(self.buf, cap=buf_cfg["capacity"])
        del buf_cfg["capacity"]

        train_ids, val_ids = set(), set()

        class SplitHook(buffer.Hook):
            def __init__(self):
                self._g = np.random.default_rng()

            def on_create(self, seq_id: int, seq: dict):
                is_val = self._g.random() < cfg.val_frac
                (val_ids if is_val else train_ids).add(seq_id)

            def on_delete(self, seq_id: int, seq: dict):
                if seq_id in val_ids:
                    val_ids.remove(seq_id)
                elif seq_id in train_ids:
                    train_ids.remove(seq_id)

        self.buf.hooks.append(SplitHook())

        self.train_sampler = buffer.Sampler()
        self.train_ds = data.Slices(
            buf=self.buf,
            sampler=self.train_sampler,
            **buf_cfg,
        )
        self.batch_iter = iter(self.train_ds)

        self.val_sampler = buffer.Sampler()

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

        self.env_step = 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.agent_step = 0

        self.buf_mtx = Lock()
        self.fwd_mtx = Lock()

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

        self.agent.mode = "train"
        should_train = self._make_until(cfg.total)

        self.opt_step = 0
        while should_train:
            self.take_env_step()

            with self.buf_mtx:
                wm_batch = next(self.batch_iter)

            with self.fwd_mtx:
                loss, metrics, states = self.wm_trainer.compute(wm_batch)

            self.wm_trainer.opt_step(loss)

            for k, v in metrics.items():
                self.exp.add_scalar(f"wm/{k}", v)

            ac_batch = {"states": states[-1], "term": wm_batch["term"]}
            with self.fwd_mtx:
                loss, metrics = self.ac_trainer.compute(ac_batch)
            self.ac_trainer.opt_step(loss)

            for k, v in metrics.items():
                self.exp.add_scalar(f"ac/{k}", v)

            self.opt_step += 1

    def _make_until(self, cfg):
        if isinstance(cfg, dict):
            n = cfg["n"]
            step_fn = lambda: getattr(self, cfg["of"])
        else:
            n = cfg
            step_fn = lambda: self.env_step
        return cron.Until(step_fn, n)

    def take_env_step(self):
        with self.fwd_mtx:
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
            diff = 1
            self.env_step += 1

        self.pbar.update(diff)
