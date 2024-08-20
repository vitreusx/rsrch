import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from lpips import LPIPS
from moviepy.editor import *
from torch import Tensor, nn

from rsrch import spaces
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.nn.utils import safe_mode
from rsrch.utils import config, cron, repro, sched
from rsrch.utils.preview import make_grid

from . import data, nets, vq
from .utils import over_seq, pass_gradient


class VqVAE_1d(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()

        self.vq = vq.VSQLayer(num_tokens=32, vocab_size=256, embed_dim=256)
        self.z_dim = self.vq.num_tokens * self.vq.embed_dim

        scale_factor = 16
        in_h, in_w = space.shape[1:]
        out_w, out_h = in_w // scale_factor, in_h // scale_factor
        embed_dim = self.z_dim // (out_w * out_h)

        params = dict(
            obs_space=space,
            conv_hidden=(2 * embed_dim // scale_factor),
            scale_factor=scale_factor,
        )

        self.encode = nn.Sequential(
            nets.Encoder_v1(**params),
            nn.Flatten(),
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, self.z_dim),
            nets.Reshape(-1, (self.vq.num_tokens, self.vq.embed_dim)),
        )

        self.decode = nn.Sequential(
            nn.Flatten(),
            nets.Reshape(-1, (embed_dim, out_h, out_w)),
            nets.Decoder_v1(**params),
        )


class VqVAE_2d(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()

        self.vq = vq.VQLayer(vocab_size=256, embed_dim=256)

        scale_factor = 8
        in_h, in_w = space.shape[1:]
        out_h, out_w = out_h // scale_factor, out_w // scale_factor

        params = dict(
            obs_space=space,
            conv_hidden=(2 * self.vq.embed_dim // scale_factor),
            scale_factor=scale_factor,
        )


class Trainer(nn.Module):
    def __init__(self, vae: VqVAE_1d | VqVAE_2d):
        super().__init__()
        self.vae = vae
        self.opt = torch.optim.Adam(self.vae.parameters(), lr=3e-4, eps=1e-5)

    def opt_step(self, batch: Tensor, opt_step: int):
        losses = {}
        coef = {"recon": 1e2}

        e = self.vae.encode(batch)

        z, z_idx = self.vae.vq(e)
        vq_embed = F.mse_loss(z, e.detach())
        vq_commit = F.mse_loss(z.detach(), e)
        losses["vq"] = 0.8 * vq_embed + 0.2 * vq_commit
        z = pass_gradient(z, e)

        recon = self.vae.decode(z)
        losses["recon"] = F.mse_loss(recon, batch)

        loss: Tensor = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10.0)
        self.opt.step()

        metrics = {f"{k}_loss": v for k, v in losses.items()}
        metrics["loss"] = loss
        return metrics

    def preview(self, batch: data.SliceBatch):
        seq_len, batch_size = batch.obs.shape[:2]
        e = over_seq(self.vae.encode)(batch.obs)
        z, z_idx = over_seq(self.vae.vq)(e)
        recon = over_seq(self.vae.decode)(z)

        frames = []
        for t in range(seq_len):
            grid = []
            for i in range(batch_size):
                orig_ = data.to_pil(batch.obs[t, i])
                recon_ = data.to_pil(recon[t, i])
                grid.append([orig_, recon_])
            grid = make_grid(grid)
            frames.append(np.asarray(grid))

        return ImageSequenceClip(frames, fps=1.0)


@dataclass
class Config:
    samples_path: Path | None
    batch_size: int
    opt_steps: int
    preview_every: int
    device: str
    seed: int


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")
    cfg = config.parse(cfg, Config)

    repro.seed_all(seed=cfg.seed, deterministic=False)

    exp = Experiment(project="vq")
    exp.boards.append(Tensorboard(exp.dir / "board"))

    device = torch.device(cfg.device)

    with open(cfg.samples_path, "rb") as f:
        samples: data.Buffer = pickle.load(f)

    obs_space = spaces.torch.Image(shape=(1, 64, 64), dtype=torch.float32)
    vae = VqVAE_1d(obs_space).to(device)
    trainer = Trainer(vae)

    opt_step = 0
    exp.register_step("opt_step", lambda: opt_step, default=True)

    ds = data.Observations(samples)
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
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_iter = iter(train_loader)

    ds = data.Slices(samples, 128)
    preview_loader = data.DataLoader(
        dataset=ds,
        batch_size=8,
        sampler=data.InfiniteSampler(ds, shuffle=False),
        collate_fn=ds.collate_fn,
    )
    preview_iter = iter(preview_loader)

    pbar = exp.pbar(desc="VQ", initial=opt_step)
    should_preview = cron.Every(lambda: opt_step, cfg.preview_every)

    while opt_step < cfg.opt_steps:
        if should_preview:
            batch: data.SliceBatch = next(preview_iter)
            batch = batch.to(device)

            vid = trainer.preview(batch)
            exp.add_video("vq/preview", vid)

        batch: Tensor = next(train_iter)
        batch = batch.to(device)

        metrics = trainer.opt_step(batch, opt_step)
        for k, v in metrics.items():
            exp.add_scalar(f"vq/{k}", v)

        opt_step += 1
        pbar.update(1)


if __name__ == "__main__":
    main()
