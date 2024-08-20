import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from moviepy.editor import *
from torch import Tensor, nn

from rsrch import distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.nn import fc
from rsrch.nn.utils import safe_mode
from rsrch.types import Tensorlike
from rsrch.utils import config, cron, repro
from rsrch.utils.preview import make_grid

from . import data, vq
from .utils import over_seq, pass_gradient


class AtariEncoder(nn.Sequential):
    def __init__(self, hidden=48, in_channels=1):
        super().__init__(
            nn.Conv2d(in_channels, hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(hidden, 2 * hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(2 * hidden, 4 * hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(4 * hidden, 8 * hidden, 4, 2),
            nn.ELU(),
            nn.Flatten(),
        )


class Reshape(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        if not isinstance(in_shape, Iterable):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        if not isinstance(out_shape, Iterable):
            out_shape = (out_shape,)
        self.out_shape = tuple(out_shape)

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class AtariDecoder(nn.Sequential):
    def __init__(self, in_features: int, hidden=48, out_channels=1):
        super().__init__(
            nn.Linear(in_features, 32 * hidden),
            Reshape(32 * hidden, (32 * hidden, 1, 1)),
            nn.ConvTranspose2d(32 * hidden, 4 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(4 * hidden, 2 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(2 * hidden, hidden, 6, 2),
            nn.ELU(),
            nn.ConvTranspose2d(hidden, out_channels, 6, 2),
        )


class Normalize(nn.Module):
    def __init__(self, enabled=True, clip: float | None = 5.0):
        super().__init__()
        self.enabled = enabled
        self.first_time = True
        self.clip = clip

        self.loc: Tensor
        self.register_buffer("loc", torch.zeros([]))
        self.inv_scale: Tensor
        self.register_buffer("inv_scale", torch.ones([]))

    def forward(self, input: Tensor):
        if self.enabled:
            if self.training and self.first_time:
                self.loc.copy_(input.mean())
                self.inv_scale.copy_(input.std().clamp_min(1e-8).reciprocal())
                self.first_time = False
            input = (input - self.loc) * self.inv_scale
            if self.clip is not None:
                input = input.clamp(-self.clip, self.clip)
        return input

    def inverse(self, input: Tensor):
        if self.enabled:
            input = input / self.inv_scale + self.loc
        return input


class SeqVAE(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()
        assert obs_space.shape == (1, 64, 64)

        self.obs_norm = Normalize(enabled=True, clip=None)
        self.obs_enc = AtariEncoder()

        with safe_mode(self.obs_enc):
            dummy = obs_space.sample()[None]
            obs_size = self.obs_enc(dummy).shape[1]

        act_size = 128
        self.act_enc = nn.Embedding(act_space.n, act_size)

        self.seq_hidden, self.seq_layers = 2048, 1
        self.fc_hidden = 512

        self._h0_model = nn.Sequential(
            AtariEncoder(),
            nn.Linear(obs_size, self.seq_hidden * self.seq_layers),
        )

        self.enc_size = obs_size + act_size
        self._enc_rnn = nn.GRU(
            input_size=self.enc_size,
            hidden_size=self.seq_hidden,
            num_layers=self.seq_layers,
        )

        self.z_tokens, self.z_vocab, self.z_embed = 8, 128, 256
        self.z_vq = vq.VSQLayer(self.z_tokens, self.z_vocab, self.z_embed)
        z_size = self.z_tokens * self.z_embed
        self._enc_proj = nn.Sequential(
            nn.Linear(self.seq_hidden, z_size),
            Reshape(-1, (self.z_tokens, self.z_embed)),
        )

        self.dec_size = z_size + act_size
        self._dec_rnn = nn.GRU(
            input_size=self.dec_size,
            hidden_size=self.seq_hidden,
            num_layers=self.seq_layers,
        )
        self._dec_proj = nn.Identity()

        self.obs_dec = AtariDecoder(self.seq_hidden)

    def rnn_init(self, obs: Tensor):
        out: Tensor = self._h0_model(obs)
        out = out.reshape(-1, self.seq_layers, self.seq_hidden)
        out = out.moveaxis(1, 0)
        return out

    def encode(
        self,
        enc_obs: Tensor,
        enc_act: Tensor,
        h0: Tensor,
    ):
        input = torch.cat((enc_act, enc_obs), -1)
        out, hx = self._enc_rnn(input, h0.contiguous())
        out = over_seq(self._enc_proj)(out)
        return out, hx

    def decode(self, z: Tensor, enc_act: Tensor, h0: Tensor):
        input = torch.cat((enc_act, z), -1)
        out, hx = self._dec_rnn(input, h0.contiguous())
        out = over_seq(self._dec_proj)(out)
        return out, hx


class Trainer(nn.Module):
    def __init__(self, vae: SeqVAE, dtype: torch.dtype):
        super().__init__()
        self.vae = vae
        self.opt = torch.optim.Adam(self.vae.parameters(), lr=3e-4, eps=1e-5)
        self.device = next(self.vae.parameters()).device
        self.dtype = dtype
        self.scaler = getattr(torch, self.device.type).amp.GradScaler()

    def autocast(self):
        return torch.autocast(self.device.type, self.dtype)

    def opt_step(self, batch: data.SliceBatch):
        losses = {}
        coef = {"recon": 1e2}

        with self.autocast():
            obs = over_seq(self.vae.obs_norm)(batch.obs)

            h0 = self.vae.rnn_init(obs[0])
            enc_obs = over_seq(self.vae.obs_enc)(obs[1:])
            enc_act = over_seq(self.vae.act_enc)(batch.act)

            out, _ = self.vae.encode(enc_obs, enc_act, h0)

            z, z_idx = self.vae.z_vq(out)
            z_vq_embed = F.mse_loss(z, out.detach())
            z_vq_commit = F.mse_loss(z.detach(), out)
            losses["vq"] = 0.8 * z_vq_embed + 0.2 * z_vq_commit
            z = pass_gradient(z, out).flatten(-2)

            out, _ = self.vae.decode(z, enc_act, h0)

            recon = over_seq(self.vae.obs_dec)(out)
            losses["recon"] = F.mse_loss(recon, obs[1:])

        loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), 10.0)
        self.opt.step()

        metrics = {f"{k}_loss": v for k, v in losses.items()}
        metrics["loss"] = loss
        return metrics

    @safe_mode()
    def preview(self, batch: data.SliceBatch):
        with self.autocast():
            obs = over_seq(self.vae.obs_norm)(batch.obs)
            orig = batch.obs[1:]

            h0 = self.vae.rnn_init(obs[0])
            enc_obs = over_seq(self.vae.obs_enc)(obs[1:])
            enc_act = over_seq(self.vae.act_enc)(batch.act)

            out, _ = self.vae.encode(enc_obs, enc_act, h0)
            z, z_idx = self.vae.z_vq(out)
            z = pass_gradient(z, out).flatten(-2)

            recon, _ = self.vae.decode(z, enc_act, h0)
            recon = over_seq(self.vae.obs_dec)(recon)
            recon = over_seq(self.vae.obs_norm.inverse)(recon)

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
    resume_from: Path | None
    batch_size: int
    slice_len: int
    seed: int
    total_steps: int
    preview_every: int
    device: str
    dtype: Literal["float32", "float16"]


class Runner:
    def main(self):
        cfg = config.load(Path(__file__).parent / "config.yml")
        cfg = config.parse(cfg, Config)

        repro.seed_all(seed=cfg.seed)

        self.exp = Experiment(project="seq_vae")
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))

        self.opt_step = 0
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        with open(cfg.samples_path, "rb") as f:
            self.samples: data.BufferData = pickle.load(f)["samples"]

        self.obs_space = spaces.torch.Image((1, 64, 64), dtype=torch.float32)
        self.act_space = spaces.torch.Discrete(18)

        vae = SeqVAE(self.obs_space, self.act_space)
        vae = vae.to(self.device)

        trainer = Trainer(vae, self.dtype)

        ds = data.Slices(data.Buffer(self.samples), cfg.slice_len)
        sampler = data.InfiniteSampler(ds, shuffle=True)
        train_loader = data.DataLoader(
            dataset=ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
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

        ds = data.Slices(data.Buffer(self.samples), 128)
        preview_loader = data.DataLoader(
            dataset=ds,
            batch_size=8,
            sampler=data.InfiniteSampler(ds, shuffle=True),
            collate_fn=ds.collate_fn,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(cfg.seed + 1),
        )
        preview_iter = iter(preview_loader)

        should_preview = cron.Every(lambda: self.opt_step, cfg.preview_every)
        pbar = self.exp.pbar(desc="SeqVAE", initial=self.opt_step)

        if cfg.resume_from is not None:
            with open(cfg.resume_from, "rb") as f:
                data_ = torch.load(f)
            trainer.load_state_dict(data_["trainer"])

            while self.opt_step < data_["opt_step"]:
                batch = next(train_iter)
                self.opt_step += 1
                pbar.update()

        while self.opt_step < cfg.total_steps:
            if should_preview:
                batch = next(preview_iter)
                batch = batch.to(self.device)

                vid = trainer.preview(batch)
                if vid is not None:
                    self.exp.add_video("wm/preview", vid)

            batch = next(train_iter)
            batch = batch.to(self.device)

            metrics = trainer.opt_step(batch)
            for k, v in metrics.items():
                self.exp.add_scalar(f"wm/{k}", v)

            self.opt_step += 1
            pbar.update()


if __name__ == "__main__":
    Runner().main()
