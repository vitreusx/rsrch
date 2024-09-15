from collections import namedtuple
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import gym

from ..common import nets
from ..common.trainer import TrainerBase
from ..common.types import Slices
from ..common.utils import autocast, find_class, tf_init
from . import rssm


@dataclass
class Config:
    @dataclass
    class KL:
        forward: bool
        balance: float
        free: float
        free_avg: bool

    encoder: dict
    decoders: dict
    rssm: rssm.Config
    opt: dict
    coef: dict[str, float]
    clip_grad: float | None
    reward_fn: Literal["id", "clip", "tanh", "sign"]
    clip_rew: tuple[float, float] | None
    kl: KL


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Box,
        rew_space: spaces.torch.Box,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space
        self.rew_space = rew_space

        self.obs_enc = nets.make_encoder(self.obs_space, **cfg.encoder)
        self.act_enc = nn.Identity()

        with safe_mode(self):
            obs: Tensor = self.obs_space.sample([1])
            obs_size: int = self.obs_enc(obs).shape[1]

            act: Tensor = self.act_space.sample([1])
            self.act_size: int = self.act_enc(act).shape[1]

        self.rssm = rssm.RSSM(cfg.rssm, obs_size, self.act_size)

        self.state_size = self.rssm.stoch_size + self.rssm.deter_size

        self.obs_dec = self._make_decoder(
            space=obs_space,
            **self.cfg.decoders["obs"],
        )
        self.reward_dec = self._make_decoder(
            space=rew_space,
            **self.cfg.decoders["reward"],
        )
        self.term_dec = self._make_decoder(
            space=spaces.torch.Discrete(2, dtype=torch.bool),
            **self.cfg.decoders["term"],
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(tf_init)

    def _make_decoder(self, space: spaces.torch.Tensor, **kwargs):
        return nn.Sequential(
            rssm.AsTensor(),
            nets.make_decoder(self.state_size, space, **kwargs),
        )

    def reset(self, obs):
        state = self.rssm.initial()
        state = state[None].expand(obs.shape[0], *state.shape)
        obs = self.obs_enc(obs.to(state.device))
        act = torch.zeros((obs.shape[0], self.act_size)).type_as(obs)
        return self.rssm.obs_step(state, act, obs)

    def obs_step(self, state, act, next_obs):
        act = self.act_enc(act)
        next_obs = self.obs_enc(next_obs)
        return self.rssm.obs_step(state, act, next_obs)

    def img_step(self, state, act):
        return self.rssm.img_step(state, act)

    def observe(
        self,
        input: tuple[Tensor, Tensor],
        h_0: list[Tensor | None],
    ):
        obs_seq, act_seq = input
        enc_obs: Tensor = over_seq(self.obs_enc)(obs_seq)
        enc_act: Tensor = over_seq(self.act_enc)(act_seq)

        is_first = np.array([x is None for x in h_0])
        enc_act[0, is_first].zero_()
        h_0 = torch.stack([self.rssm.initial() if x is None else x for x in h_0])

        return self.rssm((enc_obs, enc_act), h_0)

    def as_tensor(self, state: rssm.State):
        return state.as_tensor()


TrainerOutput = namedtuple("TrainerOutput", ("loss", "metrics", "states", "h_n"))


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        compute_dtype=None,
    ):
        super().__init__(
            clip_grad=cfg.clip_grad,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg

        if cfg.reward_fn == "clip":
            clip_low, clip_high = cfg.clip_rew
            self.reward_space = spaces.torch.Box((), low=clip_low, high=clip_high)
            self._reward_fn = lambda r: torch.clamp(r, *cfg.clip_rew)
        elif cfg.reward_fn in ("sign", "tanh"):
            self.reward_space = spaces.torch.Box((), low=-1.0, high=1.0)
            self._reward_fn = torch.sign if cfg.reward_fn == "sign" else torch.tanh
        elif cfg.reward_fn == "id":
            self.reward_space = spaces.torch.Tensor((), dtype=torch.float32)
            self._reward_fn = lambda r: r

    def setup(self, wm: WorldModel | nn.Sequential):
        self.wm = wm
        self.opt = self._make_opt()

    def _make_opt(self):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        return cls(self.wm.parameters(), **cfg)

    def compute(
        self,
        seq: Slices,
        h_0: list[Tensor | None],
    ):
        mets, losses = {}, {}

        with self.autocast():
            reward = self._reward_fn(seq.reward)

            out, h_n = self.wm.observe((seq.obs, seq.act), h_0)
            states, post, prior = out

            kl_loss, kl_value = self._kl_loss(post, prior, **vars(self.cfg.kl))
            losses["kl"] = kl_loss

            obs_dist = over_seq(self.wm.obs_dec)(states)
            if isinstance(obs_dist, dict):
                for name, dist in obs_dist.items():
                    losses[name] = -dist.log_prob(seq.obs[name]).mean()
                obs_loss = sum(losses[name].detach() for name in obs_dist)
            else:
                losses["obs"] = -obs_dist.log_prob(seq.obs).mean()

            rew_dist = over_seq(self.wm.reward_dec)(states)
            rew_loss = -rew_dist.log_prob(reward)
            is_first = np.array([x is None for x in h_0])
            rew_loss[0, is_first].zero_()
            losses["reward"] = rew_loss.mean()

            term_dist = over_seq(self.wm.term_dec)(states)
            losses["term"] = -term_dist.log_prob(seq.term).mean()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        with torch.no_grad():
            with self.autocast():
                mets = {}

                mets["kl_value"] = kl_value.mean()
                mets["prior_ent"] = prior.entropy().mean()
                mets["post_ent"] = post.entropy().mean()
                mets["reward_mean"] = reward.mean()
                mets["reward_std"] = reward.std()
                mets["reward_pred_mean"] = rew_dist.mean.mean()
                mets["reward_pred_std"] = rew_dist.mean.std()

                mets["loss"] = loss.detach()
                for k, v in losses.items():
                    mets[f"{k}_loss"] = v.detach()
                if "obs_loss" in locals():
                    mets["obs_loss"] = obs_loss

        return TrainerOutput(loss, mets, states.detach(), h_n.detach())

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

    def reset(self):
        def _reset(module):
            if isinstance(module, WorldModel):
                module.reset_parameters()

        self.wm.apply(_reset)


AsTensor = rssm.AsTensor
