import math
import os
import pickle
from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from fast_pytorch_kmeans import KMeans
from moviepy.editor import *
from PIL import Image
from scipy.optimize import linprog
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.hub.rl import env
from rsrch.nn import dist_head as dh
from rsrch.nn.utils import safe_mode
from rsrch.rl import gym
from rsrch.rl.data import rollout
from rsrch.types import Tensorlike
from rsrch.utils import config, cron, repro
from rsrch.utils.preview import make_grid

from . import data, dreamer, lm, nets, vq
from .tokenizer import Tokenizer, VisTrainer
from .utils import InputSpec, over_seq, pass_gradient


def gae_adv_est(
    r: Tensor,
    v: Tensor,
    cont: Tensor,
    gamma: float,
    gae_lambda: float,
):
    r, v = r * cont[1:], v * cont
    delta = (r + gamma * v[1:]) - v[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(r))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = v[:-1] + adv
    return adv, ret


@dataclass
class TokenizerConfig:
    enabled: bool
    pretrained: Path | None
    samples_path: Path | None
    batch_size: int
    opt_steps: int
    preview_every: int


@dataclass
class LMConfig:
    pretrained: Path | None
    batch_size: int
    slice_len: int
    opt_steps: int
    preview_every: int


@dataclass
class ACConfig:
    batch_size: int
    prefix_len: int
    horizon: int
    gamma: float
    gae_lambda: float
    update_epochs: int
    update_batch: int
    adv_norm: bool
    clip_coef: float
    clip_vloss: bool
    v_coef: float
    ent_coef: float
    lr: float
    eps: float
    clip_grad: float | None
    opt_steps: int
    clip_rew: bool
    test_every: int
    test_envs: int
    test_episodes: int


@dataclass
class Config:
    seed: int
    env: env.Config
    tokenizer: TokenizerConfig
    lm: LMConfig
    ac: ACConfig


class Runner:
    def main(self):
        self.cfg = config.load(Path(__file__).parent / "config.yml")
        self.cfg = config.parse(self.cfg, Config)

        repro.fix_seeds(seed=self.cfg.seed, deterministic=False)

        self.exp = Experiment(project="wmx")
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))
        self.opt_step = 0
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.device = torch.device("cuda")

        self.load_samples()
        if self.cfg.tokenizer.enabled:
            self.train_tokenizer()
            self.encode_samples()
        self.train_lm()

    def load_samples(self):
        with open(self.cfg.samples_path, "rb") as f:
            self.samples: data.Buffer = pickle.load(f)

        self.spec = InputSpec(
            obs=self.samples.obs_space,
            act=self.samples.act_space,
            rew=spaces.torch.Box((), -torch.inf, +torch.inf),
            term=spaces.torch.Discrete(2, dtype=torch.bool),
        )

    def train_tokenizer(self):
        cfg = self.cfg.tokenizer

        self.tok = Tokenizer(self.spec.obs, self.spec.act)
        self.tok = self.tok.to(self.device)

        if cfg.pretrained is not None:
            with open(cfg.pretrained, "rb") as f:
                state = torch.load(f, map_location="cpu")
            self.tok.load_state_dict(state["tokenizer"])
            return

        ds = data.Episodes(self.samples)
        rew = torch.tensor([r for ep in ds for r in ep.reward])
        rew = rew.to(self.device)
        self.tok.rew.fit(rew)

        trainer = VisTrainer(self.tok.obs)

        ds = data.Observations(self.samples)
        train_loader = data.DataLoader(
            dataset=ds,
            batch_size=cfg.batch_size,
            sampler=data.InfiniteSampler(ds, shuffle=True),
            num_workers=os.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=ds.collate_fn,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        train_iter = iter(train_loader)

        ds = data.Slices(self.samples, 128)
        preview_loader = data.DataLoader(
            dataset=ds,
            batch_size=8,
            sampler=data.InfiniteSampler(ds, shuffle=True),
            collate_fn=ds.collate_fn,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )
        preview_iter = iter(preview_loader)

        self.opt_step = 0
        should_preview = cron.Every(lambda: self.opt_step, cfg.preview_every)
        pbar = self.exp.pbar(desc="Tokenizer", initial=self.opt_step)

        while self.opt_step < cfg.opt_steps:
            batch: Tensor = next(train_iter)
            batch = batch.to(self.device)

            ret = trainer.opt_step(batch)
            for k, v in ret.items():
                self.exp.add_scalar(f"tok/{k}", v)

            if should_preview:
                batch: Tensor = next(preview_iter)
                batch = batch.to(self.device)
                batch = batch.obs

                vid = trainer.preview(batch)
                self.exp.add_video("tok/preview", vid)

            self.opt_step += 1
            pbar.update(1)

        ckpt_name = f"tokenizer.pth"
        ckpt_path = self.exp.dir / "ckpts" / ckpt_name
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "wb") as f:
            state = {
                "tokenizer": self.tok.state_dict(),
                "opt_step": self.opt_step,
            }
            torch.save(state, f)

    def encode_samples(self):
        path = self.cfg.tokenizer.samples_path
        if path is not None:
            with open(path, "rb") as f:
                self.samples = pickle.load(f)
            self.spec = self.tok.spec
            return

        tok_samples = data.Buffer(
            obs_space=self.tok.obs.space,
            act_space=self.tok.act.space,
        )

        episodes = data.Episodes(self.samples)
        episodes = self.exp.pbar(
            episodes,
            desc="Encoding Samples",
            total=len(episodes),
        )

        self.tok.eval()
        for ep in episodes:
            ep_id = None

            with torch.inference_mode():
                obs = ep.obs.to(self.device)
                obs = self.tok.obs.encode(obs)
                obs = obs.cpu().numpy()
                act = ep.act.to(self.device)
                act = self.tok.act.encode(act)
                act = act.cpu().numpy()
                rew = ep.reward.to(self.device)
                rew = self.tok.rew.encode(rew)
                rew = rew.cpu().numpy()

            for t in range(0, len(ep.act)):
                ep_id = tok_samples.push(
                    ep_id,
                    data.rl.Step(
                        obs=obs[t],
                        act=act[t],
                        next_obs=obs[t + 1],
                        reward=rew[t],
                        term=ep.term and t == len(ep.act) - 1,
                    ),
                )

        path = self.exp.dir / "samples" / "encoded.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(tok_samples, f)

        self.samples = tok_samples
        self.spec = self.tok.spec

    def _fixed_lm_sampler(self, ds: data.SlicesLM):
        # This piece of arcane code does following:
        #
        #   The goal is to adjust sampling probabilities of slices for LM
        # training so that (a) episode ends occur sufficiently often, (b) each
        # type of reward token occurs sufficiently often.
        #   First, let's see how many of each reward and term token does every
        # slice in the dataset have. We'll store it in the array A of shape (N, F).
        # Now, let p denote probabilities for each slice. Then, A^T p denotes
        # expected number of each token.
        #   The specific variant we take is to provide a hard lower bound on
        # the probability value for each slice, as well as for the episode ends.
        # Then, we optimize the minimum expected value out of any of the reward
        # tokens. The reason being that it's fairly easy to ensure the former
        # are satisfied, and some types of reward tokens can be rare, which makes
        # providing any explicit bound for them an issue.
        #
        #   This takes the form of a following LP:
        #
        #     Maximize r_0
        #         s.t. A^T p >= [r_0, ..., r_0, t_0]^T
        #                  p >= [p_0, ..., p_0]^T
        #              1^T p == 1
        #
        #   The rest of the code fits this LP into the form
        # `scipy.optimize.linprog` can digest.

        num_rew_tokens = self.tokenizer.rew.vocab_size
        N, R, F = len(ds), num_rew_tokens, num_rew_tokens + 1

        A = np.zeros((N, F))
        for idx, seq in enumerate(ds):
            r = np.asarray(seq.reward).ravel()
            np.add.at(A[idx], r, 1)
            A[idx, -1] = seq.term

        min_term_pr = 0.1
        min_slice_pr = 0.1 / len(ds)
        max_slice_pr = 1.0 / len(ds)

        A_ub = np.zeros((F, N + 1))
        A_ub[:, :N] = -A.T
        A_ub[:R, N:] = 1
        b_ub = np.zeros([F])
        b_ub[R:] = -min_term_pr

        A_eq = np.concatenate([np.ones(N), np.zeros(1)])[None]
        b_eq = np.array([1])

        c = np.zeros([N + 1])
        c[N:] = -1

        while True:
            bounds = [(min_slice_pr, max_slice_pr)] * N
            bounds += [(0, None)]

            r = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
            if r.success:
                break

            max_slice_pr *= 2.0

        slice_pr = r.x[:N]
        return data.WeightedRandomSampler(
            weights=slice_pr,
            num_samples=len(slice_pr),
        )

    def train_lm(self):
        cfg = self.cfg.lm

        self.tok.eval()

        self.lm = lm.WorldModel(self.tok.spec)
        self.lm = self.lm.to(self.device)

        if cfg.pretrained is not None:
            with open(cfg.pretrained, "rb") as f:
                state = torch.load(f, map_location="cpu")
            self.lm.load_state_dict(state["lm"])
            self.opt_step += cfg.opt_steps
            return

        trainer = lm.Trainer(self.lm)

        ds = data.SlicesLM(self.samples, cfg.slice_len)
        sampler = data.InfiniteSampler(ds, shuffle=True)
        train_loader = data.DataLoader(
            dataset=ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            collate_fn=ds.collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(self.cfg.seed),
            prefetch_factor=2,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )
        train_iter = iter(train_loader)

        ds = data.SlicesLM(self.samples, 128)
        sampler = data.InfiniteSampler(ds, shuffle=True)
        preview_loader = data.DataLoader(
            dataset=ds,
            batch_size=8,
            sampler=sampler,
            collate_fn=ds.collate_fn,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )
        preview_iter = iter(preview_loader)

        self.opt_step = 0
        should_preview = cron.Every(lambda: self.opt_step, cfg.preview_every)
        pbar = self.exp.pbar(desc="LM", initial=self.opt_step)

        total_steps = self.cfg.lm.opt_steps
        while self.opt_step < total_steps:
            batch: data.SliceBatch = next(train_iter)
            batch = batch.to(self.device)

            metrics = trainer.opt_step(batch)
            for k, v in metrics.items():
                self.exp.add_scalar(f"lm/{k}", v)

            if should_preview:
                batch = next(preview_iter)
                batch = batch.to(self.device)

                vid = trainer.preview(batch, self.tok)
                if vid is not None:
                    self.exp.add_video("lm/preview", vid)

            self.opt_step += 1
            pbar.update()

        ckpt_path = self.exp.dir / "ckpts" / "lm.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "wb") as f:
            state = {
                "lm": self.lm.state_dict(),
                "opt_step": self.opt_step,
            }
            torch.save(state, f)


if __name__ == "__main__":
    Runner().main()
