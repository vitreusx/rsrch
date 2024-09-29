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
from rsrch.nn import dh
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
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("loc", torch.zeros(shape))
        self.register_buffer("scale_inv", torch.ones(shape))
        self.first_run = True

    def forward(self, x: Tensor):
        if self.first_run:
            self.loc = x.mean(0)
            self.scale_inv = x.std(0).clamp_min(1e-5).reciprocal()
        return (x - self.loc) * self.scale_inv

    def inverse(self, x: Tensor):
        return (x / self.scale_inv) + self.loc


class MinMaxScaler(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("loc", torch.zeros(shape))
        self.register_buffer("scale_inv", torch.ones(shape))
        self.initialized = False

    def forward(self, x: Tensor):
        if self.training and not self.initialized:
            xmin, xmax = x.amin(0), x.amax(0)
            self.loc = xmin
            self.scale_inv = (xmax - xmin).clamp_min(1e-5).reciprocal()
        return (x - self.loc) * self.scale_inv

    def inverse(self, x: Tensor):
        return (x / self.scale_inv) + self.loc


class TokenSeqDist(Tensorlike, D.Distribution):
    def __init__(self, *, logits: Tensor):
        Tensorlike.__init__(self, logits.shape[:-2])
        self.event_shape = (logits.shape[-2],)
        self.num_tokens, self.vocab_size = logits.shape[-2:]
        self.ind_rv = self.register("ind_rv", D.Categorical(logits=logits))

    def sample(self, sample_shape=()):
        idx = self.ind_rv.sample(sample_shape)
        value = F.one_hot(idx, self.vocab_size)
        value = value.type_as(self.ind_rv.logits)
        return value

    def rsample(self, sample_shape=()):
        value = self.sample(sample_shape)
        logits = self.ind_rv.logits.expand_as(value)
        value = pass_gradient(value, logits)
        return value


@D.register_kl(TokenSeqDist, TokenSeqDist)
def _(p: TokenSeqDist, q: TokenSeqDist):
    return D.kl_divergence(p.ind_rv, q.ind_rv)


class VAE(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()
        assert space.shape == (1, 64, 64)

        self.encode = AtariEncoder()

        with safe_mode(self):
            x = space.sample()[None]
            enc_x = self.encode(x)
            enc_size = enc_x.shape[1]

        self.z_dim = 1024
        self.encode = nn.Sequential(
            self.encode,
            dh.Normal(enc_size, (self.z_dim,)),
        )

        self.decode = AtariDecoder(self.z_dim)


class Trainer(nn.Module):
    def __init__(self, vae: VAE):
        super().__init__()
        self.vae = vae
        self.opt = torch.optim.Adam(self.vae.parameters(), lr=3e-4, eps=1e-5)

        self.prior = D.Normal(
            loc=torch.zeros([self.vae.z_dim], device=self.device),
            scale=torch.ones([self.vae.z_dim], device=self.device),
            event_dims=1,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def opt_step(self, batch: Tensor):
        losses = {}
        coef = {"recon": 1.0, "prior": 1e-5}

        z_dist = self.vae.encode(batch)
        losses["prior"] = D.kl_divergence(z_dist, self.prior).mean()

        z_samp = z_dist.rsample()
        recon: Tensor = self.vae.decode(z_samp)
        losses["recon"] = F.mse_loss(recon, batch)

        loss: Tensor = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics = {f"{k}_loss": v for k, v in losses.items()}
        metrics["loss"] = loss
        return metrics

    def preview(self):
        with safe_mode(self):
            z_samp = self.prior.sample((16,))
            recon = self.vae.decode(z_samp)

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

        self.exp = Experiment(project="vae")
        self.exp.boards.append(Tensorboard(self.exp.dir / "board"))

        self.opt_step = 0
        self.exp.register_step("opt_step", lambda: self.opt_step, default=True)

        self.device = torch.device("cuda")

        with open(cfg.samples_path, "rb") as f:
            self.samples: data.Buffer = pickle.load(f)

        self.obs_space = spaces.torch.Image((1, 64, 64), dtype=torch.float32)

        vae = VAE(self.obs_space)
        vae = vae.to(self.device)

        trainer = Trainer(vae)

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
        pbar = self.exp.pbar(desc="VAE", initial=self.opt_step)

        while self.opt_step < cfg.total_steps:
            if should_sample:
                img = trainer.preview()
                self.exp.add_image("vae/samples", img)

            batch = next(train_iter)
            batch = batch.to(self.device)

            metrics = trainer.opt_step(batch)
            for k, v in metrics.items():
                self.exp.add_scalar(f"vae/{k}", v)

            self.opt_step += 1
            pbar.update()


if __name__ == "__main__":
    Runner().main()
