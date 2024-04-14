import os
import pickle
from dataclasses import dataclass
from functools import cache, wraps
from numbers import Number
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as tv_F
from moviepy.editor import ImageSequenceClip
from PIL import Image
from torch import Tensor, nn

from rsrch import distributions as D
from rsrch import spaces
from rsrch._exp import Experiment, board
from rsrch.distributions.utils import sum_rightmost
from rsrch.nn import dist_head as dh
from rsrch.nn.utils import infer_ctx
from rsrch.rl.data import EpisodeBuffer, SliceBuffer, Step
from rsrch.rl.data.types import Seq, SliceBatch, StepBatch
from rsrch.types.shared import make_shared
from rsrch.utils import cron, data
from rsrch.utils.preview import make_grid

from .vq import VQLayer, VSQLayer, pass_gradient


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
            nn.Flatten(),
        ]

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
    def __init__(
        self,
        input_space: spaces.torch.Image,
        conv_hidden: int,
    ):
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
    def __init__(
        self,
        output_space: spaces.torch.Image,
        conv_hidden: int,
    ):
        super().__init__()

        conv_w, conv_h = output_space.width // 16, output_space.height // 16
        self.in_shape = (4 * conv_hidden, conv_w, conv_h)
        self.in_features = int(np.prod(self.in_shape))

        self.main = nn.Sequential(
            ResBlock(4 * conv_hidden),
            ResBlock(4 * conv_hidden),
            nn.ConvTranspose2d(4 * conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * conv_hidden, 2 * conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * conv_hidden, conv_hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(conv_hidden, output_space.num_channels, 4, 2, 1),
        )

    def forward(self, input: Tensor):
        input = input.reshape(len(input), *self.in_shape)
        out = self.main(input)
        return D.Dirac(out, 3)


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

        self.encoder = VisEncoder2(input_space, conv_hidden=32)
        self.decoder = VisDecoder2(input_space, conv_hidden=32)

        with infer_ctx(self.encoder):
            enc_out = self.encoder(input_space.sample()[None])
            embed_dim = enc_out.shape[1]

        num_embed = 32
        self.vq = VQLayer(num_embed=num_embed, embed_dim=embed_dim)


class VQProj(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_tokens: int,
        embed_dim: int,
        input_norm=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.fc = nn.Linear(in_features, embed_dim * num_tokens)
        self.input_norm = input_norm
        if input_norm:
            self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, input: Tensor):
        output = self.fc(input)
        if self.input_norm:
            output = output.reshape(-1, self.embed_dim, self.num_tokens)
            output = self.norm(output)
            output = output.swapaxes(-2, -1)
        else:
            output = output.reshape(-1, self.num_tokens, self.embed_dim)
        return output


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.seq_hidden, self.fc_hidden = 1024, 256
        self.num_tokens, self.vocab_size, self.embed_dim = 32, 32, 128

        dummy_obs = obs_space.sample((1,))
        dummy_act = act_space.sample((1,))

        vis_enc = VisEncoder2(obs_space, conv_hidden=32)
        with infer_ctx(vis_enc):
            out_features = int(np.prod(vis_enc(dummy_obs).shape[1:]))

        self.init_enc = nn.Sequential(
            vis_enc,
            nn.Flatten(),
            nn.Linear(out_features, self.seq_hidden),
        )

        self.obs_enc = self.init_enc
        self.act_enc = nn.Embedding(act_space.n, 64)

        with infer_ctx(self.obs_enc, self.act_enc):
            obs_out = self.obs_enc(dummy_obs)[0]
            obs_dim = obs_out.shape[-1]
            act_out = self.act_enc(dummy_act)[0]
            act_dim = act_out.shape[-1]

        self.seq = nn.GRU(obs_dim + act_dim, self.seq_hidden)

        self.vq_proj = VQProj(self.seq_hidden, self.num_tokens, self.embed_dim)

        # self.vq = VQLayer(
        #     num_embed=self.num_embed,
        #     embed_dim=self.embed_dim,
        #     dim=-1,
        # )

        self.vq = VSQLayer(
            num_tokens=self.num_tokens,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )

        vis_dec = VisDecoder2(obs_space, conv_hidden=32)
        self.obs_dec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_tokens * self.embed_dim, vis_dec.in_features),
            vis_dec,
        )

        self.reward_dec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_tokens * self.embed_dim, self.fc_hidden),
            nn.ReLU(),
            dh.Dirac(self.fc_hidden, ()),
        )

        self.cont_dec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_tokens * self.embed_dim, self.fc_hidden),
            nn.ReLU(),
            dh.Bernoulli(self.fc_hidden),
        )

    def pred(self, z_t, enc_act):
        fc_x = torch.cat((z_t.flatten(1), enc_act), 1)
        logits = self.pred_fc(fc_x)
        logits = logits.reshape(-1, self.num_tokens, self.vq.num_embed)
        return D.Categorical(logits=logits, event_dims=1)


def data_loss(dist: D.Distribution, value: Tensor):
    if isinstance(dist, D.Dirac):
        return F.mse_loss(dist.value, value)
    else:
        return -dist.log_prob(value).mean()


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
        item.reward = item.reward.sign().float()
        item.term = float(item.term)
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
        seq_len, batch_size = args[0].shape[:2]
        y = _func(*(x.flatten(0, 1) for x in args), **kwargs)
        if isinstance(y, tuple):
            xs = []
            for x in y:
                if len(x.shape) > 0:
                    x = x.reshape(seq_len, batch_size, *x.shape[1:])
                xs.append(x)
            y = tuple(xs)
        else:
            if len(y.shape) > 0:
                y = y.reshape(seq_len, batch_size, *y.shape[1:])
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
    obs_space = spaces.torch.Image(input_shape, dtype=torch.float32)
    act_space = spaces.torch.Discrete(ds.env_f.act_space.n)

    wm = WorldModel(obs_space, act_space).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)

    should_train = cron.Until(lambda: opt_step, cfg.num_steps)
    should_log_img = cron.Every2(lambda: opt_step, cfg.log_images_every)

    pbar = exp.pbar(total=cfg.num_steps, desc="VQ")

    coef = {"obs": 256.0, "pred": 1e-2}

    while True:
        if not should_train:
            break

        # Train step
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        batch = SliceBatch(
            obs=batch.obs.to(device),
            act=batch.act.to(device),
            reward=batch.reward.to(device),
            term=batch.term.to(device),
        )

        losses = {}

        h_t = over_seq(wm.init_enc)(batch.obs)  # [L, N, H]
        h_proj = over_seq(wm.vq_proj)(h_t)
        h_proj = h_proj.reshape(*h_proj.shape[:-2], wm.num_tokens, wm.embed_dim)

        z_t, z_idx, losses["vq"] = over_seq(wm.vq)(h_proj, compute_loss=True)
        z_t = pass_gradient(z_t, h_proj)

        obs_rv = over_seq(wm.obs_dec)(z_t)
        losses["obs"] = data_loss(obs_rv, batch.obs)

        # h0 = wm.init_enc(batch.obs[0])  # [N, D, #VQ]
        # h0 = h0.flatten(1).unsqueeze(0)  # [1, N, D * #VQ = H]
        # h0 = h0.repeat(wm.seq.num_layers, 1, 1)  # [#L, N, H]

        # enc_obs = over_seq(wm.obs_enc)(batch.obs[1:])  # [L, N, D_o]
        # enc_act = over_seq(wm.act_enc)(batch.act)  # [L, N, D_a]
        # x_t = torch.cat((enc_act, enc_obs), -1)  # [L, N, D_x]
        # out, _ = wm.seq(x_t, h0)  # [L, N, H]
        # h_t = torch.cat((h0[-1][None], out), 0)  # [L + 1, N, H]

        # h_proj = over_seq(wm.vq_proj)(h_t)
        # h_proj = h_proj.reshape(*h_proj.shape[:-2], wm.embed_dim, wm.num_vq)

        # z_t, z_idx, losses["vq"] = wm.vq(h_proj, compute_loss=True)
        # z_t = pass_gradient(z_t, h_proj)

        # obs_rv = over_seq(wm.obs_dec)(z_t)
        # losses["obs"] = data_loss(obs_rv, batch.obs)

        # rew_rv = over_seq(wm.reward_dec)(z_t[1:])
        # losses["rew"] = data_loss(rew_rv, batch.reward)

        # cont_rv = over_seq(wm.cont_dec)(z_t[1:])
        # losses["cont"] = data_loss(cont_rv, 1.0 - batch.term)

        # pred_rv = over_seq(wm.pred)(z_t[:-1], enc_act)
        # losses["pred"] = data_loss(pred_rv, z_idx[1:])

        wm_loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        wm_opt.step()
        opt_step += 1
        pbar.update()

        exp.add_scalar("train/loss", wm_loss)
        for k, v in losses.items():
            exp.add_scalar(f"train/{k}_loss", v)

        vq_usage = (wm.vq._no_use_streak == 0).sum()
        exp.add_scalar("train/vq_usage", vq_usage)

        if should_log_img:
            frames = [
                np.asarray(obs_to_img(obs).convert("RGB")) for obs in batch.obs[:, 0]
            ]
            orig_vid = ImageSequenceClip(frames, fps=4.0)
            exp.add_video("train/orig_vid", orig_vid)

            frames = [
                np.asarray(obs_to_img(obs).convert("RGB")) for obs in obs_rv[:, 0].mode
            ]
            recon_vid = ImageSequenceClip(frames, fps=4.0)
            exp.add_video("train/recon_vid", recon_vid)


if __name__ == "__main__":
    main()
