import os
import pickle
from dataclasses import dataclass
from functools import cache, wraps
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch import Tensor, nn

from rsrch import distributions as D
from rsrch import spaces
from rsrch._exp import Experiment, board
from rsrch.nn.utils import infer_ctx
from rsrch.rl.data import EpisodeBuffer, SliceBuffer, Step
from rsrch.rl.data.types import Seq, SliceBatch, StepBatch
from rsrch.types.shared import make_shared
from rsrch.utils import cron, data
from rsrch.utils.preview import make_grid

from .vq import VQLayer, pass_gradient


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


class VisEncoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Image,
        conv_hidden: int,
        out_features=None,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        assert (space.width, space.height) == (64, 64)
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [
            nn.Conv2d(space.shape[0], conv_hidden, 4, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.Conv2d(conv_hidden, 2 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.Conv2d(2 * conv_hidden, 4 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.Conv2d(4 * conv_hidden, 8 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(8 * conv_hidden),
            # At this point the size is [2, 2]
        ]

        if out_features is not None:
            layers.extend(
                [
                    nn.Flatten(),
                    nn.Linear(32 * conv_hidden, out_features),
                ]
            )

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        super().__init__(*layers)


class VisDecoder(nn.Module):
    def __init__(
        self,
        space: spaces.torch.Image,
        conv_hidden: int,
        state_dim=None,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        assert tuple(space.shape[-2:]) == (64, 64)

        self.state_dim = state_dim
        if state_dim is not None:
            self.fc = nn.Linear(state_dim, 8 * conv_hidden)

        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [
            nn.ConvTranspose2d(8 * conv_hidden, 4 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.ConvTranspose2d(4 * conv_hidden, 2 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.ConvTranspose2d(2 * conv_hidden, conv_hidden, 6, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.ConvTranspose2d(conv_hidden, space.shape[0], 6, 2),
        ]

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        self.convt = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        if self.state_dim is not None:
            x = self.fc(x)
            x = x.reshape(*x.shape, 1, 1)
        x = self.convt(x)
        x = x.reshape(len(x), self.out_values, -1, *x.shape[2:])
        return D.Dirac(x, 3)


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


class VisEncoder2(nn.Sequential):
    def __init__(self, input_space: spaces.torch.Image, conv_hidden: int):
        super().__init__(
            nn.Conv2d(input_space.num_channels, conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(2 * conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(2 * conv_hidden, 4 * conv_hidden, 4, 2, 1),
            ResBlock(4 * conv_hidden),
            ResBlock(4 * conv_hidden),
        )


class VisDecoder2(nn.Module):
    def __init__(self, input_space: spaces.torch.Image, conv_hidden: int):
        super().__init__()
        self.main = nn.Sequential(
            ResBlock(4 * conv_hidden),
            ResBlock(4 * conv_hidden),
            nn.ConvTranspose2d(4 * conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * conv_hidden, conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_hidden, input_space.num_channels, 4, 2, 1),
        )

    def forward(self, input: Tensor):
        out = self.main(input)
        return D.Dirac(out, 3)


class Reshape(nn.Module):
    def __init__(self, in_shape: tuple[int, ...], out_shape: tuple[int, ...]):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class VQModel(nn.Module):
    def __init__(self, input_shape=(3, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        input_space = spaces.torch.Image(input_shape, dtype=torch.float32)

        # self.encoder = nn.Sequential(
        #     # VisEncoder(input_space, out_features=state_dim, conv_hidden=32),
        #     # Reshape((state_dim,), (embed_dim, num_embed)),
        #     VisEncoder(input_space, conv_hidden=32),
        # )
        # self.decoder = nn.Sequential(
        #     # Reshape((embed_dim, num_embed), (state_dim,)),
        #     VisDecoder(input_space, state_dim=state_dim, conv_hidden=32),
        # )

        self.encoder = VisEncoder2(input_space, conv_hidden=64)
        self.decoder = VisDecoder2(input_space, conv_hidden=64)

        with infer_ctx(self.encoder):
            enc_out = self.encoder(input_space.empty()[None])
            embed_dim = enc_out.shape[1]

        num_embed = 256
        self.vq = VQLayer(num_embed=num_embed, embed_dim=embed_dim)

        ...


def get_recon_loss(x_hat: D.Distribution, x: Tensor):
    if isinstance(x_hat, D.Dirac):
        return F.mse_loss(x_hat.value, x)
    else:
        return -x_hat.log_prob(x).mean()


class SampleImages(data.Dataset):
    def __init__(self, path: str, slice_len=32):
        super().__init__()
        self.slice_len = slice_len
        with open(path, "rb") as f:
            data = pickle.load(f)
            ep_buf, self.env_f = data["sample_buf"], data["env_f"]

        self.buf = SliceBuffer.from_episodes(
            ep_buf=ep_buf,
            slice_len=slice_len,
        )

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, idx):
        item = self.buf[idx]
        obs = item.obs
        obs = obs[:, -1]
        obs = torch.as_tensor(obs / 255.0, dtype=torch.float32)
        obs = tv_F.resize(obs, (64, 64))
        item.obs = obs
        item.act = torch.as_tensor(item.act, dtype=torch.long)
        item.reward = torch.as_tensor(item.reward, dtype=torch.float32)
        item.term = bool(item.term)
        return item


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


def default_collate_fn(batch: list[Step] | list[Seq]):
    if isinstance(batch[0], Step):
        obs = torch.stack([x.obs for x in batch], 0)
        act = torch.stack([x.act for x in batch], 0)
        next_obs = torch.stack([x.next_obs for x in batch], 0)
        reward = torch.tensor([x.reward for x in batch], device=obs.device)
        term = torch.tensor([x.term for x in batch], device=obs.device)
        trunc = None
        if batch[0].trunc is not None:
            trunc = torch.tensor([x.trunc for x in batch], device=obs.device)
        return StepBatch(obs, act, next_obs, reward, term, trunc)
    else:
        obs = torch.stack([x.obs for x in batch], 1)
        act = torch.stack([x.act for x in batch], 1)
        reward = torch.stack([x.reward for x in batch], 1)
        term = torch.tensor([x.term for x in batch], device=obs.device)
        return SliceBatch(obs, act, reward, term)


@cache
def _over_seq(_func):
    @wraps(_func)
    def _lifted(*args, **kwargs):
        batch_size = args[0].shape[1]
        y = _func(*(x.flatten(0, 1) for x in args), **kwargs)
        y = y.reshape(-1, batch_size, *y.shape[1:])
        return y

    return _lifted


def over_seq(_func):
    """Transform a function that operates on batches (N, ...) to operate on
    sequences (L, N, ...). The reshape takes place for positional arguments."""
    return _over_seq(_func)


@dataclass
class Config:
    device = "cuda"
    batch_size = 64
    slice_len = 32
    train_workers = os.cpu_count()
    val_workers = os.cpu_count()
    prefetch_factor = 2
    lr = 2e-4
    num_steps = int(250e3)
    sample_pkl: str = "runs/rainbow/ALE-Alien-v5__2024-04-08_08-30-11/samples.pkl"
    log_images_every = 256


def main():
    cfg = Config()

    exp = Experiment(
        project="seq_vq",
        board=board.Tensorboard,
    )

    opt_step = 0
    exp.register_step("opt_step", lambda: opt_step, default=True)

    device = torch.device(cfg.device)

    ds = SampleImages(path=cfg.sample_pkl, slice_len=cfg.slice_len)

    loader = data.DataLoader(
        dataset=ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        # num_workers=cfg.train_workers,
        # prefetch_factor=cfg.prefetch_factor,
        collate_fn=default_collate_fn,
    )

    batch_iter = iter(loader)

    def obs_to_img(obs: Tensor):
        img = obs[0].float().cpu().clamp(0.0, 1.0)
        return tv_F.to_pil_image(img)

    input_shape = ds[0].obs.shape[1:]
    model = VQModel(input_shape).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    should_train = cron.Until(lambda: opt_step, cfg.num_steps)
    should_log_img = cron.Every2(lambda: opt_step, cfg.log_images_every)

    pbar = exp.pbar(total=cfg.num_steps, desc="VQ")

    while True:
        if not should_train:
            break

        # Train step
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        slices = batch
        obs, act = slices.obs, slices.act  # [L, N, ...]
        obs, act = obs.to(device), act.to(device)

        obs = obs.flatten(0, 1)
        obs = obs[np.random.randint(len(obs), size=(128,))]
        e = model.encoder(obs)
        if opt_step == 0:
            model.vq.init(e)
        z_idx, z, vq_loss = model.vq(e, compute_loss=True)
        y = model.decoder(pass_gradient(z, e))
        recon_loss = get_recon_loss(y, obs)
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
                orig_img = obs_to_img(obs[idx])
                recon_img = obs_to_img(y[idx].mode)
                samples[idx] = [orig_img, recon_img]
            exp.add_image("train/samples", make_grid(samples))


if __name__ == "__main__":
    main()
