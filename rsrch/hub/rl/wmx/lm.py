from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import *
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.types import Tensorlike
from rsrch.utils.preview import make_grid

from . import data
from .tokenizer import InputSpec, Tokenizer
from .utils import over_seq, pass_gradient


class Tag(IntEnum):
    OBS = 0
    ACT = 1


class WorldModel(nn.Module):
    def __init__(self, input_spec: InputSpec):
        super().__init__()
        self.spec = input_spec

        self.tag_size = 64
        self._tag_emb = nn.Embedding(2, self.tag_size)

        self.token_size = 256
        self._obs_emb = nn.Embedding(
            self.spec.obs.vocab_size,
            self.token_size,
        )
        self._act_emb = nn.Embedding(
            self.spec.act.vocab_size,
            self.token_size,
        )

        self.input_size = self.token_size + self.tag_size

        self.seq_hidden, self.seq_layers = 1024, 2
        self.seq_model = nn.GRU(
            self.input_size,
            self.seq_hidden,
            self.seq_layers,
        )

        self.proj = nn.Linear(self.seq_hidden, self.spec.obs.vocab_size)

    def obs_emb(self, obs: Tensor) -> Tensor:
        out = self._obs_emb(obs)
        tag = self._tag_emb.weight[Tag.OBS]
        tag = tag.expand(*out.shape[:-1], self.tag_size)
        out = torch.cat((out, tag), -1)
        return out

    def act_emb(self, act: Tensor) -> Tensor:
        out = self._act_emb(act)
        tag = self._tag_emb.weight[Tag.ACT]
        tag = tag.expand(*out.shape[:-1], self.tag_size)
        out = torch.cat((out, tag), -1)
        return out

    def pack(
        self,
        obs: Tensor,
        act: Tensor,
        *,
        encoded=False,
    ):
        if encoded:
            enc_obs, enc_act = obs, act
        else:
            enc_obs = self.obs_emb(obs)  # [L+1, B, N_o, D]
            enc_act = self.act_emb(act)  # [L, B, N_a, D]

        seq_len = enc_obs.shape[0] + enc_act.shape[0]
        batch_size = enc_obs.shape[1]

        seq_x = torch.cat((act, enc_obs[1:]), 2)
        seq_x = seq_x.moveaxis(2, 1).flatten(0, 1)  # [L*(N_o+N_a), B, D]
        seq_x = torch.cat((enc_obs[0].moveaxis(1, 0), seq_x), 0)

        obs_tag = torch.empty(
            enc_obs.shape[:-1],
            dtype=torch.long,
            device=seq_x.device,
        )
        obs_tag.fill_(Tag.OBS)

        act_tag = torch.empty(
            enc_act.shape[:-1],
            dtype=torch.long,
            device=seq_x.device,
        )
        act_tag.fill_(Tag.ACT)

        tag = torch.cat((act_tag, obs_tag[1:]), 2)
        tag = tag.moveaxis(2, 1).flatten(0, 1)
        tag = torch.cat((obs_tag[0].moveaxis(1, 0), tag), 0)

        return seq_x, tag


class Trainer(nn.Module):
    def __init__(self, wm: WorldModel):
        super().__init__()
        self.wm = wm
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=3e-4, eps=1e-5)

    def opt_step(self, batch: data.SliceBatch):
        enc_obs = self.wm.obs_emb(batch.obs)
        enc_act = self.wm.act_emb(batch.act)
        seq_x, tag = self.wm.pack(enc_obs, enc_act, encoded=True)

        out, _ = self.wm.seq_model(seq_x)

        is_next_obs = torch.where(tag[:-1] == Tag.OBS)
        logits = self.wm.proj(out[:-1][is_next_obs])
        targets = batch.obs.moveaxis(2, 1).flatten(0, 1)[1:].ravel()
        loss = F.cross_entropy(logits, targets)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics = {"loss": loss}
        return metrics

    def preview(self, batch: data.SliceBatch, tok: Tokenizer):
        ...
