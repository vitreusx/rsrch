from functools import cached_property, partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import data

from . import config, nets, rssm


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Space,
        act_space: spaces.torch.Space,
        cfg: config.WorldModel,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg
        self.dtype = dtype

        self.obs_enc = self._make_net(cfg.encoders.obs)(obs_space)
        self.act_enc = self._make_net(cfg.encoders.act)(act_space)
        self.act_dec = self._make_decoder(self.act_enc)

        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            obs_size: int = self.obs_enc(obs).shape[1]
            act: Tensor = act_space.sample([1])
            act_size: int = self.act_enc(act).shape[1]

        self.rssm = rssm.EnsembleRSSM(cfg.rssm, obs_size, act_size)
        self.state_size = self.rssm.stoch_size + self.rssm.deter_size

        self.obs_dec = self._make_net(cfg.decoders.obs)(
            self.state_size,
            obs_space,
        )

        if cfg.reward_fn == "clip":
            rew_space = spaces.torch.Box((), *cfg.clip_rew)
        elif cfg.reward_fn == "sign":
            rew_space = spaces.torch.OneOf(torch.tensor([-1.0, 0.0, 1.0]))
        elif cfg.reward_fn == "tanh":
            rew_space = spaces.torch.Box((), -1.0, +1.0)
        else:
            rew_space = spaces.torch.Box((), -torch.inf, +torch.inf)

        self.reward_dec = self._make_net(cfg.decoders.reward)(
            self.state_size,
            rew_space,
        )

        term_space = spaces.torch.Discrete(2, dtype=torch.bool)
        self.term_dec = self._make_net(cfg.decoders.term)(self.state_size, term_space)

        self.opt = self._make_opt(cfg.opt)(self.parameters())

    @cached_property
    def scaler(self):
        return getattr(torch, self.device.type).amp.GradScaler()

    @cached_property
    def device(self):
        return next(self.parameters()).device

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def _make_net(self, cfg: dict) -> Callable[..., nn.Module]:
        cfg = {**cfg}
        cls = getattr(nets, cfg["$type"])
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_opt(self, cfg: dict) -> Callable[..., torch.optim.Optimizer]:
        cfg = {**cfg}
        cls = getattr(torch.optim, cfg["$type"])
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_decoder(self, encoder: nn.Module):
        if isinstance(encoder, nets.DiscreteEncoder):
            return encoder.inverse
        elif isinstance(encoder, nets.Identity):
            return encoder
        else:
            raise ValueError(f"Cannot create a decoder for {type(encoder)}")

    def opt_step(self, batch: data.SliceBatch, state: rssm.State, is_first: Tensor):
        loss, ret = self._loss_fn(batch, state, is_first)

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if self.cfg.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.cfg.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()

        return ret

    def _loss_fn(self, batch: data.SliceBatch, state: rssm.State, is_first: Tensor):
        losses = {}

        with self.autocast():
            enc_obs = over_seq(self.obs_enc)(batch.obs)
            enc_act = over_seq(self.act_enc)(batch.act).type_as(enc_obs)

            post, prior = self.rssm.observe(enc_obs, enc_act, state, is_first)
            kl_loss, kl_value = self._kl_loss(post, prior, **self.cfg.kl)
            losses["kl"] = kl_loss

            state = post.rsample()
            state_t = state.to_tensor()

            obs_out = over_seq(self.obs_dec)(state_t)
            obs = batch.obs
            losses["obs"] = self._recon_loss(obs_out, obs)

            reward_out = over_seq(self.reward_dec)(state_t)
            reward = self.reward_fn(batch.reward)
            losses["reward"] = self._recon_loss(reward_out, reward)

            term_out = self.term_dec(state_t[-1])
            term = batch.term.float()
            losses["term"] = self._recon_loss(term_out, term)

            coef = self.cfg.coef
            loss: Tensor = sum(
                coef[k] * v if k in coef else v for k, v in losses.items()
            )

            mets = {
                "loss": loss,
                "kl_value": kl_value.mean(),
                "prior_ent": prior.entropy().mean(),
                "post_ent": post.entropy().mean(),
                **{f"{k}_loss": v for k, v in losses.items()},
            }

        ret = mets, state[-1].detach(), batch.term[-1]

        return loss, ret

    def reward_fn(self, reward: Tensor):
        if self.cfg.reward_fn == "sign":
            reward = reward.sign()
        elif self.cfg.reward_fn == "clip":
            reward = reward.clamp(*self.cfg.clip_rew)
        elif self.cfg.reward_fn == "tanh":
            reward = reward.tanh()
        return reward

    def _kl_loss(
        self,
        post: rssm.StateDist,
        prior: rssm.StateDist,
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

    def _recon_loss(self, input: Tensor | D.Distribution, target: Tensor):
        if isinstance(input, Tensor):
            return F.mse_loss(input, target)
        else:
            return -input.log_prob(target).mean()
