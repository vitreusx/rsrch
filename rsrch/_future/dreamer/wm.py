from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq

from .amp import autocast
from .utils import find_class


@dataclass
class Config:
    opt: dict
    decoders: dict


class RSSM:
    initial: Tensor

    def reset(self, obs):
        ...

    def step(self, state, act, next_obs):
        ...

    def observe(self, h0, act_seq, obs_seq):
        ...


class WorldModel:
    obs_enc: nn.Module
    act_enc: nn.Module
    act_dec: nn.Module
    rssm: RSSM

    def reset(self, obs):
        obs = self.obs_enc(obs)
        return self.rssm.reset(obs)

    def step(self, state, act, next_obs):
        act = self.act_enc(act)
        next_obs = self.obs_enc(next_obs)
        return self.rssm.step(state, act, next_obs)


class Trainer(nn.Module):
    def __init__(self, wm: WorldModel, cfg: Config, device=None, dtype=None):
        self.wm = wm
        self.cfg = cfg

        self.decoders: nn.ModuleDict

        self.opt = self._make_opt()

    def _make_opt(self):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        return cls(self.wm.parameters(), **cfg)

    def train_loss(self, batch: dict):
        return self._loss_fn(batch, True)

    def val_loss(self, batch: dict):
        return self._loss_fn(batch, False)

    def _loss_fn(self, batch: dict, with_metrics=True):
        mets, losses = {}, {}
        rssm = self.wm.rssm

        with autocast():
            enc_obs: Tensor = over_seq(self.wm.obs_enc)(batch["obs"])
            enc_act: Tensor = over_seq(self.wm.act_enc)(batch["act"])

            is_first = np.array([x is None for x in batch["start"]])
            enc_act[0, is_first].zero_()
            starts = torch.stack([x or rssm.initial for x in batch["start"]])

            states, post, prior = rssm.observe(enc_obs, enc_act, starts)
            if with_metrics:
                mets["prior_ent"] = prior.detach().entropy().mean()
                mets["post_ent"] = post.detach().entropy().mean()

            kl_loss, kl_value = self._kl_loss(post, prior, **self.cfg.kl)
            losses["kl"] = kl_loss
            if with_metrics:
                mets["kl_value"] = kl_value.detach().mean()

            for name, decoder in self.decoders.items():
                recon: D.Distribution = over_seq(decoder)(states)
                losses[name] = -recon.log_prob(batch[name]).mean()

            if with_metrics:
                for k, v in losses:
                    mets[f"{k}_loss"] = v.detach()

        coef = self.cfg.coef
        loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        return loss, mets, states

    def _kl_loss(
        self,
        post: D.Distribution,
        prior: D.Distribution,
        forward: bool,
        balance: float,
        free: float,
        free_avg: bool,
    ):
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else 1.0 - balance
        if balance == 0.5:
            value = D.kl_divergence(lhs, rhs)
            loss = value.clamp_min(free).mean()
        else:
            value_lhs = value = D.kl_divergence(lhs, rhs.detach())
            value_rhs = D.kl_divergence(lhs.detach(), rhs)
            if free_avg:
                loss_lhs = value_lhs.mean().clamp_min(free)
                loss_rhs = value_rhs.mean().clamp_min(free)
            else:
                loss_lhs = value_lhs.clamp_min(free).mean()
                loss_rhs = value_rhs.clamp_min(free).mean()
            loss = mix * loss_lhs + (1.0 - mix) * loss_rhs
        return loss, value
