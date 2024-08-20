import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import *
from torch import Tensor, nn

from rsrch import distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.nn import dist_head as dh
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


class AE(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()
        assert space.shape == (1, 64, 64)

        self.encoder = AtariEncoder()

        with safe_mode(self):
            x = space.sample()[None]
            enc_x = self.encoder(x)
            self.z_dim = enc_x.shape[1]

        self.decoder = AtariDecoder(self.z_dim)


class Trainer(nn.Module):
    def __init__(self, ae: AE):
        super().__init__()
        self.ae = ae
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=3e-4, eps=1e-5)

    @property
    def device(self):
        return next(self.parameters()).device

    def opt_step(self, batch: Tensor):
        coef = {}
        losses = {}

        z = self.ae.encoder(batch)
        recon: Tensor = self.ae.decoder(z)
        losses["recon"] = F.mse_loss(recon, batch)

        loss: Tensor = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics = {f"{k}_loss": v for k, v in losses.items()}
        metrics["loss"] = loss
        return metrics

    @safe_mode()
    def preview(self, batch: Tensor):
        z = self.ae.encoder(batch[:16])
        recon = self.ae.decoder(z)

        grid = np.empty((8, 2), dtype=object)
        idxes = np.arange(16).reshape((8, 2))
        for i, j in np.ndindex(grid.shape):
            grid[i, j] = data.to_pil(recon[idxes[i, j]])
        grid = grid.tolist()
        return make_grid(grid)


@dataclass
class Config:
    samples_path: Path
    batch_size: int
    seed: int
    total_steps: int
    sample_every: int


class Runner:
    def main(self):
        cfg = config.load(Path(__file__).parent / "config.yml")
        cfg = config.parse(cfg, Config)

        repro.seed_all(seed=cfg.seed, deterministic=False)

        self.exp = Experiment(project="ae")
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))

        self.opt_step = 0
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.device = torch.device("cuda")

        with open(cfg.samples_path, "rb") as f:
            self.samples: data.Buffer = pickle.load(f)

        self.obs_space = spaces.torch.Image((1, 64, 64), dtype=torch.float32)

        ae = AE(self.obs_space).to(self.device)
        trainer = Trainer(ae)

        ds = data.Observations(self.samples)
        collate_fn = ds.collate_fn

        sampler = data.InfiniteSampler(ds, shuffle=True)
        train_loader = data.DataLoader(
            dataset=ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=repro.worker_init_fn,
            generator=torch.Generator().manual_seed(cfg.seed),
            prefetch_factor=2,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )
        train_iter = iter(train_loader)

        should_sample = cron.Every(lambda: self.opt_step, cfg.sample_every)
        pbar = self.exp.pbar(desc="AE", initial=self.opt_step)

        while self.opt_step < cfg.total_steps:
            if should_sample:
                batch = next(train_iter)
                batch = batch.to(self.device)

                img = trainer.preview(batch)
                self.exp.add_image("ae/samples", img)

            batch = next(train_iter)
            batch = batch.to(self.device)

            metrics = trainer.opt_step(batch)
            for k, v in metrics.items():
                self.exp.add_scalar(f"ae/{k}", v)

            self.opt_step += 1
            pbar.update()


if __name__ == "__main__":
    Runner().main()
