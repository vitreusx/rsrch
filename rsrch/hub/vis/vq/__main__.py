from dataclasses import dataclass
import os
from typing import Literal
import numpy as np
import torch
from torch import nn, Tensor
from rsrch import distributions as D
import torch.nn.functional as F
from rsrch.nn.utils import infer_ctx
from rsrch.utils import data
from rsrch._exp import Experiment, board
import torchvision.transforms as T
import torchvision.transforms.functional as tv_F
from rsrch.data.imagenet import ImageNet
from rsrch.data.cifar import CIFAR10, CIFAR100
from rsrch.utils import cron
from rsrch.utils.preview import make_grid
from fast_pytorch_kmeans import KMeans

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PassGradientOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, to: Tensor):
        return value

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return None, grad_output


def pass_gradient(value: Tensor, to: Tensor):
    return PassGradientOp.apply(value, to)


class VQLayer(nn.Module):
    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        commit_coef=0.25,
        replace_after=20,
    ):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commit_coef = commit_coef

        self.replace_after = replace_after
        if self.replace_after is not None:
            self.last_used: Tensor
            self.register_buffer("last_used", torch.zeros(num_embed, dtype=torch.long))

        self.codebook = nn.Parameter(torch.randn(self.num_embed, self.embed_dim))

    def init(self, input: Tensor):
        """Initialize (or reset) the codebook, using K-Means on a given input.
        :param input: Tensor of shape (N, D, ...)."""

        input = input.moveaxis(1, -1).flatten(0, -2)  # [N, D, ...] -> [*, D]
        if len(input) < self.num_embed:
            num_rep = (self.num_embed + len(input) - 1) // len(input)
            input = input.repeat(num_rep, 1)

        kmeans = KMeans(
            n_clusters=self.num_embed,
            init_method="kmeans++",
            minibatch=1024,
        )
        kmeans.fit(input)

        with torch.no_grad():
            self.codebook.copy_(kmeans.centroids)

    def replace_unused(self, input: Tensor, idxes: Tensor):
        """Replace unused code vectors by randomly sampled vectors from the
        input.
        :param input: Input tensor of shape (N, D).
        :param idxes: Index tensor of shape (*)."""

        is_used = torch.zeros(self.num_embed, device=idxes.device, dtype=torch.bool)
        is_used.index_fill_(0, idxes.ravel(), True)
        self.last_used[is_used] = 0
        self.last_used[~is_used] += 1
        to_replace = torch.where(self.last_used >= self.replace_after)[0]
        if len(to_replace) > 0:
            samp_idxes = torch.randint(0, len(input), (len(to_replace),))
            with torch.no_grad():
                self.codebook[to_replace] = input[samp_idxes]

    def forward(self, input: Tensor, compute_loss=False):
        # input: [N, D, ...]

        codebook = self.codebook
        input_ = input.moveaxis(1, -1)  # [N, D, ...] -> [N, ..., D]
        batch_shape = input_.shape[:-1]  # [N, ...]
        input_ = input_.flatten(0, -2)  # [N, ..., D] -> [*, D]
        dists = torch.cdist(input_[None], codebook[None])[0]  # [*, #E]
        idxes = dists.argmin(-1).reshape(batch_shape)  # [N, ...]

        embed = codebook[idxes]  # [N, ..., D]
        embed = embed.moveaxis(-1, 1)  # [N, ..., D] -> [N, D, ...]

        if self.replace_after is not None:
            self.replace_unused(input_, idxes)

        if compute_loss:
            codebook_loss = F.mse_loss(embed.detach(), input)
            commitment_loss = F.mse_loss(embed, input.detach())
            vq_loss = codebook_loss + self.commit_coef * commitment_loss
            return idxes, embed, vq_loss
        else:
            return idxes, embed


class ResBlock(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, 1),
        )

    def forward(self, input: Tensor):
        return input + self.net(input)


class DecoderHead(nn.Module):
    def __init__(self, input_shape: tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.in_features = max(input_shape[0], 2)
        else:
            self.in_features = int(np.prod(input_shape))

    def forward(self, input: Tensor):
        if len(self.input_shape) == 3:
            num_channels = self.input_shape[0]
            if num_channels == 1:
                return D.Bernoulli(logits=input.squeeze(1))
            else:
                return D.Dirac(input, 3)
        else:
            input = input.reshape(len(input), *self.input_shape)
            return D.Dirac(input, len(self.input_shape))


class VQModel(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        embed_dim = 256

        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim , embed_dim, 4, 2, 1),
            ResBlock(embed_dim),
            ResBlock(embed_dim),
        )

        self.vq = VQLayer(num_embed=512, embed_dim=embed_dim)

        decoder_head = DecoderHead(input_shape)
        self.decoder = nn.Sequential(
            ResBlock(embed_dim),
            ResBlock(embed_dim),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, decoder_head.in_features, 4, 2, 1),
            decoder_head,
        )


def get_recon_loss(x_hat: D.Distribution, x: Tensor):
    if isinstance(x_hat, D.Dirac):
        return F.mse_loss(x_hat.value, x)
    else:
        return -x_hat.log_prob(x)


@dataclass
class Config:
    device = "cuda"
    batch_size = 128
    train_workers = os.cpu_count()
    val_workers = os.cpu_count()
    prefetch_factor = 2
    lr = 2e-4
    num_steps = int(250e3)
    val_every = int(5e3)
    dataset: Literal["cifar-10", "cifar-100", "imagenet"] = "imagenet"
    data_root = "data/imagenet-100"
    log_images_every = 256


class reservoir(list):
    def __init__(self, maxlen=None):
        super().__init__()
        self.maxlen = maxlen

    def try_insert(self, item=None) -> int | None:
        """Try to add an item to the reservoir. If not provided, the index
        is returned - this is useful if constructing the item is expensive."""

        idx = len(self)
        if self.maxlen is not None:
            if len(self) >= self.maxlen:
                idx = np.random.randint(len(self) + 1)
                if idx >= self.maxlen:
                    return None
        if idx >= len(self):
            self.append(None)
        if item is not None:
            self[idx] = item
        else:
            return idx


def main():
    cfg = Config()

    exp = Experiment(
        project="vq",
        board=[board.Tensorboard],
    )

    opt_step = 0
    exp.register_step("opt_step", lambda: opt_step, default=True)

    device = torch.device(cfg.device)

    norm = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    norm_inv = T.Normalize(
        mean=-np.array(norm.mean) / np.array(norm.std),
        std=1.0 / np.array(norm.std),
    )

    crop_size = {"imagenet": 128, "cifar-10": 64, "cifar-100": 64}[cfg.dataset]

    train_t = T.Compose(
        [
            # T.RandomResizedCrop(224),
            T.RandomResizedCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ]
    )

    val_t = T.Compose(
        [
            # T.Resize(256),
            # T.CenterCrop(224),
            T.Resize(int(crop_size * 256 / 224)),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            norm,
        ]
    )

    if cfg.dataset == "cifar-10":
        train_ds = CIFAR10(
            root=cfg.data_root,
            train=True,
            download=True,
            transform=train_t,
        )
        val_ds = CIFAR10(
            root=cfg.data_root,
            train=False,
            download=False,
            transform=val_t,
        )
    if cfg.dataset == "cifar-100":
        train_ds = CIFAR100(
            root=cfg.data_root,
            train=True,
            download=True,
            transform=train_t,
        )
        val_ds = CIFAR100(
            root=cfg.data_root,
            train=False,
            download=False,
            transform=val_t,
        )
    elif cfg.dataset == "imagenet":
        train_ds = data.Pipeline(
            ImageNet(
                root=cfg.data_root,
                split="train",
                img_transform=train_t,
            ),
            lambda item: (item["image"], item["label"]),
        )
        val_ds = data.Pipeline(
            ImageNet(
                root=cfg.data_root,
                split="val",
                img_transform=val_t,
            ),
            lambda item: (item["image"], item["label"]),
        )

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.train_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    train_iter = iter(train_loader)

    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.val_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    def val_img_to_pil(img: Tensor):
        img = img.float().cpu()
        img = norm_inv(img).clamp(0.0, 1.0)
        return tv_F.to_pil_image(img)

    input_shape = val_ds[0][0].shape
    model = VQModel(input_shape).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    should_train = cron.Until(lambda: opt_step, cfg.num_steps)
    should_val = cron.Every2(lambda: opt_step, cfg.val_every)
    should_log_img = cron.Every2(lambda: opt_step, cfg.log_images_every)

    pbar = exp.pbar(total=cfg.num_steps, desc="VQ")

    while True:
        if should_val:
            # Val epoch
            model.eval()
            with torch.no_grad():
                samples = reservoir(maxlen=8)

                val_loss, val_n = 0.0, 0
                for batch in exp.pbar(val_loader, desc="Val"):
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)
                    batch_n = len(images)

                    e: Tensor = model.encoder(images)
                    z_idx, z, vq_loss = model.vq(e, compute_loss=True)
                    y: D.Distribution = model.decoder(z)

                    recon_loss = get_recon_loss(y, images)
                    loss = recon_loss + vq_loss

                    val_loss += loss * batch_n
                    val_n += batch_n

                    for idx in range(batch_n):
                        samp_idx = samples.try_insert()
                        if samp_idx is None:
                            continue

                        orig_img = val_img_to_pil(images[idx])
                        recon_img = val_img_to_pil(y[idx].mode)
                        samples[samp_idx] = [orig_img, recon_img]

                exp.add_image("val/samples", make_grid(samples))

                val_loss /= val_n
                exp.add_scalar("val/loss", val_loss)

            model.train()

        if not should_train:
            break

        # Train step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        e = model.encoder(images)
        if opt_step == 0:
            model.vq.init(e)
        z_idx, z, vq_loss = model.vq(e, compute_loss=True)
        y = model.decoder(pass_gradient(z, e))
        recon_loss = get_recon_loss(y, images)
        loss = recon_loss + vq_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        opt_step += 1
        pbar.update()

        exp.add_scalar("train/loss", loss)
        for k in ("recon", "vq"):
            exp.add_scalar(f"train/{k}_loss", locals()[f"{k}_loss"])

        is_used = torch.zeros(model.vq.num_embed, dtype=torch.bool, device=device)
        is_used.index_fill_(0, z_idx.ravel(), True)
        vq_usage = is_used.long().sum()
        exp.add_scalar("train/vq_usage", vq_usage)

        if should_log_img:
            samples = [None for _ in range(8)]
            for idx in range(len(samples)):
                orig_img = val_img_to_pil(images[idx])
                recon_img = val_img_to_pil(y[idx].mode)
                samples[idx] = [orig_img, recon_img]
            exp.add_image("train/samples", make_grid(samples))

if __name__ == "__main__":
    main()
