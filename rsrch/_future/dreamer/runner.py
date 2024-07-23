from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

from rsrch import spaces
from rsrch._future import rl
from rsrch._future.rl import buffer
from rsrch.exp import Experiment, timestamp
from rsrch.utils import repro

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

    seed: int
    device: str
    compute_dtype: Literal["float16", "float32"]

    debug: Debug
    env: rl.api.Config
    wm: wm.Config
    ac: ac.Config
    train: Train


# self.reward_fn = reward_fn
# self.clip_rew = clip_rew
# if self.reward_fn == "clip":
#     clip_low, clip_high = self.clip_rew
#     self.reward_space = spaces.torch.Box((), low=clip_low, high=clip_high)
# elif self.reward_fn in ("sign", "tanh"):
#     self.reward_space = spaces.torch.Box((), low=-1.0, high=1.0)
# elif self.reward_fn == "id":
#     self.reward_space = spaces.torch.Space((), dtype=torch.float32)


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def main(self):
        ...

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

    def _setup_nets(self):
        # We need the trainer to deduce the reward space, even if we don't
        # run the training.
        self.wm_trainer = wm.Trainer(self.cfg.wm, self.compute_dtype)

        obs_space, act_space = self.env_api.obs_space, self.env_api.act_space
        rew_space = self.wm_trainer.reward_space
        self.wm = wm.WorldModel(self.cfg.wm, obs_space, act_space, rew_space)
        self.wm = self.wm.to(self.device)

        self.actor = ac.Actor(self.wm, self.cfg.ac)
        self.actor = self.actor.to(self.device)

    def _setup_train(self):
        cfg = self.cfg.train

        buf_cfg = {**cfg.buffer}

        buf: buffer.Buffer = buffer.SizeLimited(
            buffer.Buffer(),
            cap=buf_cfg["capacity"],
        )
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

        buf.hooks.append(SplitHook())

        self.train_sampler = buffer.Sampler()
        self.train_ds = data.Slices(
            buf=buf,
            sampler=self.train_sampler,
            **buf_cfg,
        )

        self.val_sampler = buffer.Sampler()

        self.wm_trainer.setup(self.wm)

        self.ac_trainer = ac.Trainer(self.cfg.ac, self.compute_dtype)
        make_critic = lambda: ac.Critic(self.wm, self.cfg.ac).to(self.device)
        self.ac_trainer.setup(self.wm, self.actor, make_critic)
