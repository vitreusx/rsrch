from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode

from . import nets, rssm
from .utils import find_class, null_ctx


@dataclass
class Config:
    encoder: dict
    decoders: dict
    rssm: rssm.Config
    opt: dict
    coef: dict
    clip_grad: float | None
    reward_fn: Literal["id", "clip", "tanh", "sign"]
    clip_rew: tuple[float, float] | None


class WorldModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: spaces.torch.Space,
        act_space: spaces.torch.Space,
        rew_space: spaces.torch.Space,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space
        self.rew_space = rew_space

        self.obs_enc = self._make_encoder(self.obs_space, **cfg.encoder)
        self.act_enc = nets.ActionEncoder(self.act_space)

        with safe_mode(self):
            obs: Tensor = self.obs_space.sample([1])
            obs_size: int = self.obs_enc(obs).shape[1]

            act: Tensor = self.act_space.sample([1])
            self.act_size: int = self.act_enc(act).shape[1]

        self.rssm = rssm.RSSM(cfg.rssm, obs_size, self.act_size)

        self.state_size = self.rssm.stoch_size + self.rssm.deter_size

        spaces_ = {
            "obs": obs_space,
            "reward": rew_space,
            "term": spaces.torch.Discrete(2),
        }

        self.decoders = nn.ModuleDict()
        for key in spaces_:
            dec = self._make_decoder(spaces_[key], **self.cfg.decoders[key])
            self.decoders[key] = dec

    def _make_encoder(self, space: spaces.torch.Space, **args):
        cls = find_class(nets, args["type"] + "_encoder")
        del args["type"]
        return cls(space, **args)

    def _make_decoder(self, space: spaces.torch.Space, **args):
        cls = find_class(nets, args["type"] + "_decoder")
        del args["type"]
        return cls(self.state_size, space, **args)

    def reset(self, obs):
        state = self.rssm.initial
        state = state[None].expand(len(obs), *state.shape)
        obs = self.obs_enc(obs.to(state.device))
        act = torch.zeros((len(obs), self.act_size)).type_as(obs)
        return self.rssm.obs_step(state, act, obs)

    def step(self, state, act, next_obs):
        act = self.act_enc(act)
        next_obs = self.obs_enc(next_obs)
        return self.rssm.obs_step(state, act, next_obs)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        compute_dtype=None,
    ):
        self.cfg = cfg
        self.compute_dtype = compute_dtype

        if cfg.reward_fn == "clip":
            clip_low, clip_high = cfg.clip_rew
            self.reward_space = spaces.torch.Box((), low=clip_low, high=clip_high)
            self._reward_fn = lambda r: np.clip(r, *cfg.clip_rew)
        elif cfg.reward_fn in ("sign", "tanh"):
            self.reward_space = spaces.torch.Box((), low=-1.0, high=1.0)
            self._reward_fn = np.sign if cfg.reward_fn == "sign" else np.tanh
        elif cfg.reward_fn == "id":
            self.reward_space = spaces.torch.Space((), dtype=torch.float32)
            self._reward_fn = lambda r: r

    def setup(self, wm: WorldModel):
        self.wm = wm
        self.opt = self._make_opt()
        self.parameters = self.wm.parameters()
        self._device = next(self.wm.parameters()).device
        self.scaler = getattr(torch, self._device.type).amp.GradScaler()

    def save(self):
        return {
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load(self, ckpt):
        self.opt.load_state_dict(ckpt["opt"])
        self.scaler.load_state_dict(ckpt["scaler"])

    def _make_opt(self):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        return cls(self.wm.parameters(), **cfg)

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self._device.type,
                dtype=self.compute_dtype,
            )

    def compute(self, batch: dict):
        mets, losses = {}, {}
        rssm = self.wm.rssm

        with self.autocast():
            enc_obs: Tensor = over_seq(self.wm.obs_enc)(batch["obs"])
            enc_act: Tensor = over_seq(self.wm.act_enc)(batch["act"])
            batch["reward"] = self._reward_fn(batch["reward"])

            is_first = np.array([x is None for x in batch["start"]])
            enc_act[0, is_first].zero_()
            starts = torch.stack([x or rssm.initial for x in batch["start"]])

            states, post, prior = rssm.observe(enc_obs, enc_act, starts)
            mets["prior_ent"] = prior.detach().entropy().mean()
            mets["post_ent"] = post.detach().entropy().mean()

            kl_loss, kl_value = self._kl_loss(post, prior, **self.cfg.kl)
            losses["kl"] = kl_loss
            mets["kl_value"] = kl_value.detach().mean()

            for name, decoder in self.decoders.items():
                recon: D.Distribution = over_seq(decoder)(states)
                losses[name] = -recon.log_prob(batch[name]).mean()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

            mets["loss"] = loss.detach()
            for k, v in losses:
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

    def opt_step(self, loss: Tensor):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if self.cfg.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=self.cfg.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()
