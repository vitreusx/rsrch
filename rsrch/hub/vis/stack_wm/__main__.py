import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import ImageSequenceClip
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.nn.utils import safe_mode
from rsrch.utils import config, cron, repro
from rsrch.utils.preview import make_grid

from . import data, vq
from .nets import *
from .utils import over_seq, pass_gradient


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        obs_space = spaces.torch.Image((4, 64, 64), torch.float32)
        act_space = spaces.torch.Discrete(18)

        self.input_norm = Normalize(obs_space.shape)

        self.obs_enc = AtariEncoder(in_channels=4)

        with safe_mode(self):
            obs = obs_space.sample([1])
            enc_obs = self.obs_enc(obs)
            self.obs_size = enc_obs.shape[1]

        self.act_size = 128
        self.act_enc = nn.Embedding(act_space.n, self.act_size)

        self.state_size = self.obs_size

        self.z_tokens, self.z_vocab, self.z_embed = 8, 64, 128
        self.pred_vq = vq.VSQLayer(self.z_tokens, self.z_vocab, self.z_embed)
        self.z_size = self.z_tokens * self.z_embed

        cond_size = self.state_size + self.act_size
        self.pred_vq_enc = nn.Sequential(
            nn.Linear(self.state_size + cond_size, self.z_size),
            nn.ELU(),
            nn.Linear(self.z_size, self.z_size),
            Reshape(-1, (self.z_tokens, self.z_embed)),
        )
        self.pred_vq_dec = nn.GRUCell(
            input_size=self.z_size + self.act_size,
            hidden_size=self.state_size,
        )

        self.obs_dec = AtariDecoder(
            self.state_size,
            out_channels=4,
        )


class Trainer(nn.Module):
    def __init__(self, wm: WorldModel, dtype: torch.dtype):
        super().__init__()
        self.wm = wm
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=3e-4, eps=1e-5)
        self.device = next(wm.parameters()).device
        self.dtype = dtype
        self.scaler = getattr(torch, self.device.type).amp.GradScaler()

    def autocast(self):
        return torch.autocast(self.device.type, self.dtype)

    def opt_step(self, batch: data.SliceBatch):
        coef = {"pred": 1e-1}

        obs = batch.obs[:-1].moveaxis(0, 1).flatten(1, 2)
        next_obs = batch.obs[1:].moveaxis(0, 1).flatten(1, 2)

        with self.autocast():
            losses = {}

            obs = self.wm.input_norm(obs)
            state = self.wm.obs_enc(obs)

            next_obs = self.wm.input_norm(next_obs)
            next_state = self.wm.obs_enc(next_obs)
            recon = self.wm.obs_dec(next_state)

            enc_act = self.wm.act_enc(batch.act[-1])

            enc_x = torch.cat((next_state, state, enc_act), 1)
            e = self.wm.pred_vq_enc(enc_x)

            # vq, vq_idx = self.wm.pred_vq(e)
            # vq_embed = F.mse_loss(vq, e.detach())
            # vq_commit = F.mse_loss(vq.detach(), e)
            # losses["vq"] = 0.8 * vq_embed + 0.2 * vq_commit
            # z = pass_gradient(vq, e).flatten(-2)
            z = e.flatten(-2)

            dec_x = torch.cat((enc_act, z), 1)
            pred = self.wm.pred_vq_dec(dec_x, state)
            recon_pred = self.wm.obs_dec(pred)

            recon_loss = F.mse_loss(recon, next_obs)
            recon_pred_loss = F.mse_loss(recon_pred, next_obs)
            pred_loss = F.mse_loss(pred, next_state.detach())
            sensitivity = ((recon_pred_loss - recon_loss) / pred_loss).detach()
            losses["future"] = recon_loss + sensitivity * pred_loss

            # pred_err = (pred - next_state).square().mean(0)
            # state_scale = state.std(0)
            # losses["pred"] = (pred_err / state_scale).mean()

            loss = sum(coef[k] * v if k in coef else v for k, v in losses.items())

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.wm.parameters(), max_norm=10.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        metrics = {}
        for k, v in losses.items():
            metrics[f"{k}_loss"] = v
        metrics["loss"] = loss

        return metrics

    @safe_mode()
    def preview(self, batch: data.SliceBatch):
        obs = orig = batch.obs

        with self.autocast():
            obs = over_seq(self.wm.input_norm)(obs)
            state = over_seq(self.wm.obs_enc)(obs)
            recon = over_seq(self.wm.obs_dec)(state)
            recon = over_seq(self.wm.input_norm.inverse)(recon)

        frames = []
        seq_len, batch_size = orig.shape[:2]
        for t in range(seq_len):
            grid = []
            for i in range(batch_size):
                orig_ = data.to_pil(orig[t, i])
                recon_ = data.to_pil(recon[t, i])
                grid.append([orig_, recon_])
            grid = make_grid(grid)
            frames.append(np.asarray(grid))

        return ImageSequenceClip(frames, fps=1.0)


@dataclass
class Config:
    samples_path: Path
    batch_size: int
    seed: int
    total_steps: int
    test_every: int
    device: str
    dtype: Literal["float32", "float16", "bfloat16"]


class Runner:
    def main(self):
        cfg = config.load(Path(__file__).parent / "config.yml")
        cfg = config.parse(cfg, Config)

        repro.seed_all(seed=cfg.seed, deterministic=False)

        self.exp = Experiment(project="stack_wm")
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))

        self.opt_step = 0
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.dtype = getattr(torch, cfg.dtype)
        self.device = torch.device(cfg.device)

        with open(cfg.samples_path, "rb") as f:
            samples_data: data.BufferData = pickle.load(f)["samples"]
            self.samples = data.Buffer(samples_data)

        stack_num = 4

        wm = WorldModel().to(self.device)
        trainer = Trainer(wm, self.dtype)

        ds = data.Slices(self.samples, slice_steps=stack_num)
        train_loader = data.DataLoader(
            dataset=ds,
            batch_size=cfg.batch_size,
            sampler=data.InfiniteSampler(ds, shuffle=True),
            collate_fn=ds.collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(cfg.seed),
            prefetch_factor=2,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )
        train_iter = iter(train_loader)

        ds = data.Slices(data.Buffer(samples_data, 4), 128)
        test_loader = data.DataLoader(
            dataset=ds,
            batch_size=8,
            sampler=data.InfiniteSampler(ds, shuffle=False),
            collate_fn=ds.collate_fn,
            drop_last=True,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(cfg.seed + 1),
            prefetch_factor=2,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )
        test_iter = iter(test_loader)

        should_test = cron.Every(lambda: self.opt_step, cfg.test_every)
        pbar = self.exp.pbar(desc="StackWM", initial=self.opt_step)

        while self.opt_step < cfg.total_steps:
            if should_test:
                batch: data.SliceBatch = next(test_iter)
                batch = batch.to(self.device)

                vid = trainer.preview(batch)
                if vid is not None:
                    self.exp.add_video(f"test/samples", vid)

            batch: data.SliceBatch = next(train_iter)
            batch = batch.to(self.device)

            metrics = trainer.opt_step(batch)
            for k, v in metrics.items():
                self.exp.add_scalar(f"train/{k}", v)

            self.opt_step += 1
            pbar.update()


if __name__ == "__main__":
    Runner().main()
