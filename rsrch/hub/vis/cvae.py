from dataclasses import dataclass
from datetime import datetime
from functools import partial

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tv_F
from torch import Tensor
from torchvision.datasets import MNIST
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch import nn
from rsrch.exp.tensorboard import Experiment
from rsrch.nn import dist_head as dh
from rsrch.nn.utils import infer_ctx
from rsrch.utils import data
from rsrch.utils.preview import make_grid


@dataclass
class Config:
    device = "cuda"
    batch_size = 64
    num_epochs = 128
    optim = dict(type="Adam")


def main():
    cfg = Config()

    exp = Experiment(
        project="cvae",
        run=f"MNIST__{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )

    train_ds = data.Pipeline(
        MNIST(root="data/MNIST", train=True, download=True),
        lambda img_c: (tv_F.to_tensor(img_c[0]), img_c[1]),
    )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    val_ds = data.Pipeline(
        MNIST(root="data/MNIST", train=False, download=True),
        lambda img_c: (tv_F.to_tensor(img_c[0]), img_c[1]),
    )

    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
    )

    device = torch.device(cfg.device)

    state_dim = 2

    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28**2, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        dh.Normal(256, [state_dim]),
    )
    encoder = encoder.to(device)

    decoder = nn.Sequential(
        nn.Linear(state_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 28**2),
        nn.Reshape((1, 28, 28)),
        dh.Bernoulli(None),
    )
    decoder = decoder.to(device)

    # encoder = nn.Sequential(
    #     nn.Conv2d(1, 32, 3, 2, 1),
    #     nn.ReLU(),
    #     nn.Conv2d(32, 64, 3, 2, 1),
    #     nn.ReLU(),
    #     nn.Flatten(),
    #     dh.Normal(64 * 7 * 7, [state_dim]),
    # )
    # encoder = encoder.to(device)

    # decoder = nn.Sequential(
    #     nn.Linear(state_dim, 64 * 7 * 7),
    #     nn.ReLU(),
    #     nn.Reshape((64, 7, 7)),
    #     nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
    #     nn.ReLU(),
    #     nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
    #     nn.ReLU(),
    #     nn.ConvTranspose2d(32, 1, 3, 1, 1, 0),
    #     dh.Bernoulli(None),
    # )
    # decoder = decoder.to(device)

    def make_optim(cfg):
        t = getattr(torch.optim, cfg["type"])
        cfg = {**cfg}
        del cfg["type"]
        return partial(t, **cfg)

    vae_params = [*encoder.parameters(), *decoder.parameters()]
    optim = make_optim(cfg.optim)(vae_params)

    prior = D.Normal(torch.zeros(state_dim), torch.ones(state_dim), 1)
    prior = prior.to(device)

    epoch, opt_step = 0, 0
    exp.register_step("epoch", lambda: epoch)
    exp.register_step("opt_step", lambda: opt_step)

    make_pbar = partial(tqdm, dynamic_ncols=True)

    epoch_pb = make_pbar(desc="cVAE", total=cfg.num_epochs)

    while True:
        with infer_ctx(encoder, decoder):
            val_loss, nitems = 0.0, 0
            grid = []
            for images, classes in make_pbar(val_loader, desc="Val"):
                c, x = classes.to(device), images.to(device)
                z_dist: D.Distribution = encoder(x)
                z_samp: Tensor = z_dist.rsample()
                x_hat_dist: D.Categorical = decoder(z_samp)
                prior_loss = D.kl_divergence(z_dist, prior).mean()
                recon_loss = -x_hat_dist.log_prob(x).mean() * 28**2
                vae_loss = prior_loss + recon_loss
                val_loss += vae_loss * len(x)
                nitems += len(x)

                for idx in range(len(x)):
                    if len(grid) < 8:
                        xi = tv_F.to_pil_image(x[idx].cpu())
                        xi_hat = x_hat_dist[idx].mean.detach()
                        xi_hat = tv_F.to_pil_image(xi_hat.cpu())
                        grid.append([xi, xi_hat])

            val_loss /= nitems
            exp.add_scalar("val/loss", val_loss, step="epoch")
            exp.add_image("val/images", make_grid(grid), step="epoch")

        if epoch >= cfg.num_epochs:
            break

        for images, classes in make_pbar(train_loader, desc="Train"):
            c, x = classes.to(device), images.to(device)
            z_dist: D.Distribution = encoder(x)
            z_samp: Tensor = z_dist.rsample()
            x_hat_dist: D.Categorical = decoder(z_samp)
            prior_loss = D.kl_divergence(z_dist, prior).mean()
            recon_loss = -x_hat_dist.log_prob(x).mean() * 28**2
            vae_loss = prior_loss + recon_loss

            optim.zero_grad(set_to_none=True)
            vae_loss.backward()
            optim.step()

            exp.add_scalar("train/loss", vae_loss, step="opt_step")
            opt_step += 1

        epoch += 1
        epoch_pb.update()


if __name__ == "__main__":
    main()
