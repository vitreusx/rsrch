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

from . import data, nets, vq
from .utils import InputSpec, over_seq, pass_gradient


class VisTokenizer1d(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()

        self.vq = vq.VSQLayer(num_tokens=32, vocab_size=32, embed_dim=256)
        self.space = spaces.torch.TokenSeq(
            num_tokens=self.vq.num_tokens,
            vocab_size=self.vq.vocab_size,
        )
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

        self._encoder = nn.Sequential(
            nets.Encoder_v1(**params),
            nn.Flatten(),
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, self.z_dim),
            nets.Reshape(-1, (self.vq.num_tokens, self.vq.embed_dim)),
        )

        self._decoder = nn.Sequential(
            nn.Flatten(),
            nets.Reshape(-1, (embed_dim, out_h, out_w)),
            nets.Decoder_v1(**params),
        )

    def encode(self, obs: Tensor, train=False):
        e: Tensor = self._encoder(obs)
        z, z_idx = self.vq(e)
        if train:
            return z, z_idx, e
        else:
            return z_idx

    def decode(self, z: Tensor, train=False):
        if not train:
            z = self.vq.gather(z)
        out = self._decoder(z)
        return out


class VisTokenizer2d(nn.Module):
    def __init__(self, space: spaces.torch.Image):
        super().__init__()
        self.embed_dim, self.vocab_size, self.scale_factor = 256, 64, 8

        x = space.sample([1])

        in_h, in_w = space.shape[1:]
        out_w, out_h = in_w // self.scale_factor, in_h // self.scale_factor
        self._dec_shape = (self.embed_dim, out_h, out_w)

        self.num_tokens = out_h * out_w
        self.space = spaces.torch.TokenSeq(self.num_tokens, self.vocab_size)

        params = dict(
            obs_space=space,
            conv_hidden=(2 * self.embed_dim // self.scale_factor),
            scale_factor=self.scale_factor,
            norm_layer=nn.BatchNorm2d,
        )

        encoder = nets.Encoder_v1(**params)
        with safe_mode(encoder):
            enc_x = encoder(x)
            enc_shape = enc_x.shape[1:]

        self.vq = vq.VQLayer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
        )

        self._encoder = nn.Sequential(
            encoder,
            nn.Conv2d(enc_shape[0], self.embed_dim, 1),
        )

        self._decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, enc_shape[0], 1),
            nets.Decoder_v1(**params),
        )

    def encode(self, obs: Tensor, train=False):
        e: Tensor = self._encoder(obs)  # [N, C, H, W]
        e = e.moveaxis(1, -1).flatten(1, 2)
        z, z_idx = self.vq(e)
        if train:
            return z, z_idx, e
        else:
            return z_idx

    def decode(self, z: Tensor, train=False):
        if not train:
            z = self.vq.gather(z)
        z = z.reshape(len(z), *self._dec_shape)
        out = self._decoder(z)
        return out


class ScalarTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 1
        self.register_buffer("embed", torch.empty([]))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # The override for .state_dict() is needed due to .embed tensor being
        # generated only during the .fit() call.
        if prefix + "embed" in state_dict:
            device = self.embed.device
            embed = state_dict[prefix + "embed"].to(device)
            self.register_buffer("embed", embed)

    @property
    def vocab_size(self):
        return len(self.embed)

    @property
    def space(self):
        return spaces.torch.TokenSeq(1, self.vocab_size)

    def fit(self, values: Tensor):
        uniq: Tensor = values.unique()
        if len(uniq) < 64:
            embed = uniq.float()
        else:
            idxes = torch.linspace(0, len(embed) - 1, 64).round().long()
            embed = embed[idxes]

        self.embed: Tensor
        self.register_buffer("embed", embed)

    def encode(self, value: Tensor):
        l1 = (value[:, None].float() - self.embed).abs()
        return l1.argmin(-1).unsqueeze(1)

    def decode(self, z: Tensor):
        return self.embed[z.squeeze(1)]


class DiscreteTokenizer(nn.Module):
    def __init__(self, space: spaces.torch.Discrete):
        super().__init__()
        self._orig_space = space
        self.space = spaces.torch.TokenSeq(1, space.n)

    def encode(self, value: Tensor):
        return value.long().unsqueeze(1)

    def decode(self, z: Tensor):
        return z.to(self._orig_space.dtype).squeeze(1)


class Tokenizer(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.obs = VisTokenizer1d(obs_space)
        self.act = DiscreteTokenizer(act_space)
        self.rew = ScalarTokenizer()
        term_space = spaces.torch.Discrete(2, torch.bool)
        self.term = DiscreteTokenizer(term_space)

    @property
    def spec(self):
        return InputSpec(
            obs=self.obs.space,
            act=self.act.space,
            rew=self.rew.space,
            term=self.term.space,
        )


class VisTrainer(nn.Module):
    def __init__(self, tok: VisTokenizer1d | VisTokenizer2d):
        super().__init__()
        self.tok = tok
        self.opt = torch.optim.Adam(self.tok.parameters(), lr=3e-4, eps=1e-5)

    def opt_step(self, batch: Tensor):
        losses: dict[str, Tensor] = {}
        coef = {"recon": 1.0e2, "vq_commit": 0.25}

        z, _, e = self.tok.encode(batch, train=True)
        losses["vq_embed"] = F.mse_loss(z, e.detach())
        losses["vq_commit"] = F.mse_loss(z.detach(), e)
        z = pass_gradient(z, e)
        recon = self.tok.decode(z, train=True)
        losses["recon"] = F.mse_loss(recon, batch)

        loss: Tensor = sum(coef.get(k, 1.0) * v for k, v in losses.items())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.tok.parameters(), 10.0)
        self.opt.step()

        ret = {f"{k}_loss": v.item() for k, v in losses.items()}
        ret["loss"] = loss.item()
        return ret

    def preview(self, batch: Tensor):
        with safe_mode(self.tok):
            z_idx = over_seq(self.tok.encode)(batch)
            recon = over_seq(self.tok.decode)(z_idx)

        seq_len, batch_size = batch.shape[:2]
        frames = []
        for t in range(seq_len):
            grid = []
            for idx in range(batch_size):
                obs_ = data.to_pil(batch[t, idx])
                recon_ = data.to_pil(recon[t, idx])
                grid.append([obs_, recon_])
            frame = np.asarray(make_grid(grid))
            frames.append(frame)

        return ImageSequenceClip(frames, fps=1.0)
