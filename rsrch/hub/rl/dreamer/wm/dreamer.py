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

from ..common import nets, rssm
from ..common.trainer import TrainerBase
from ..common.utils import autocast, find_class, tf_init


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
        act_space: spaces.torch.Tensor,
        rew_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space
        self.rew_space = rew_space

        self.obs_enc = nets.make_encoder(self.obs_space, **cfg.encoder)
        self.act_enc = nets.ActionEncoder(self.act_space)

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

    def step(self, state, act, next_obs):
        act = self.act_enc(act)
        next_obs = self.obs_enc(next_obs)
        return self.rssm.obs_step(state, act, next_obs)

    def img_step(self, state, enc_act):
        return self.rssm.img_step(state, enc_act)

    def observe(self, obs, act, start):
        enc_obs: Tensor = over_seq(self.obs_enc)(obs)
        enc_act: Tensor = over_seq(self.act_enc)(act)

        is_first = np.array([x is None for x in start])
        enc_act[0, is_first].zero_()
        starts = torch.stack([self.rssm.initial() if x is None else x for x in start])

        return self.rssm.observe(enc_obs, enc_act, starts)

    def as_tensor(self, state: rssm.State):
        return state.as_tensor()


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

    def setup(self, wm: WorldModel):
        self.wm = wm
        self.parameters = [*self.wm.parameters()]
        self.opt = self._make_opt()

    def _make_opt(self):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        return cls(self.parameters, **cfg)

    def compute(self, obs, act, reward, term, start):
        mets, losses = {}, {}

        with self.autocast():
            reward = self._reward_fn(reward)

            states, post, prior = self.wm.observe(obs, act, start)

            kl_loss, kl_value = self._kl_loss(post, prior, **vars(self.cfg.kl))
            losses["kl"] = kl_loss

            obs_dist = over_seq(self.wm.obs_dec)(states)
            if isinstance(obs_dist, dict):
                for name, dist in obs_dist.items():
                    losses[name] = -dist.log_prob(obs[name]).mean()
                mets["obs_loss"] = sum(losses[name].detach() for name in obs_dist)
            else:
                losses["obs"] = -obs_dist.log_prob(obs).mean()

            rew_dist = over_seq(self.wm.reward_dec)(states)
            rew_loss = -rew_dist.log_prob(reward)
            is_first = np.array([x is None for x in start])
            rew_loss[0, is_first].zero_()
            losses["reward"] = rew_loss.mean()

            term_dist = over_seq(self.wm.term_dec)(states)
            losses["term"] = -term_dist.log_prob(term).mean()

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

        return loss, mets, states.detach()

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


AsTensor = rssm.AsTensor


class Agent(gym.VecAgentWrapper):
    def __init__(
        self,
        agent: gym.VecAgent,
        wm: WorldModel,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(agent)
        self.wm = wm
        self.compute_dtype = compute_dtype
        self._state = None

    @cached_property
    def device(self):
        return next(self.wm.parameters()).device

    @contextmanager
    def compute_ctx(self):
        if self.wm.training:
            self.wm.eval()

        with torch.no_grad():
            with autocast(self.device, self.compute_dtype):
                yield

    def reset(self, idxes, obs):
        obs = obs.to(self.device)
        with self.compute_ctx():
            state = self.wm.reset(obs)
        super().reset(idxes, state)
        if self._state is None:
            self._state = state.clone()
        else:
            self._state[idxes] = state.type_as(self._state)

    def policy(self, idxes):
        enc_act = super().policy(idxes)
        act = self.wm.act_enc.inverse(enc_act)
        return act

    def step(self, idxes, act: Tensor, next_obs: Tensor):
        act, next_obs = act.to(self.device), next_obs.to(self.device)
        with self.compute_ctx():
            state = self.wm.step(self._state[idxes], act, next_obs)
        state = state.type_as(self._state)
        super().step(idxes, act, state)
        self._state[idxes] = state
