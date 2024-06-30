from functools import cached_property, partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import data

from . import config, nets, rssm, rssm_opt


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Space,
        act_space: spaces.torch.Space,
        cfg: config.WorldModel,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg
        self.device, self.dtype = device, dtype

        self.obs_enc = self._make_encoder(cfg.encoders.obs)(obs_space)
        self.act_enc = self._make_encoder(cfg.encoders.act)(act_space)
        self.act_dec = self._make_inv(self.act_enc)

        with safe_mode(self):
            obs: Tensor = obs_space.sample([1])
            obs_size: int = self.obs_enc(obs).shape[1]
            act: Tensor = act_space.sample([1])
            act_size: int = self.act_enc(act).shape[1]

        # self.rssm = rssm.EnsembleRSSM(cfg.rssm, obs_size, act_size)
        self.rssm = rssm_opt.EnsembleRSSM(cfg.rssm, obs_size, act_size).to(device)
        self.rssm: rssm_opt.EnsembleRSSM = torch.jit.script(self.rssm)

        self.state_size = self.rssm.stoch_size + self.rssm.deter_size

        self.obs_dec = self._make_decoder(cfg.decoders.obs)(
            in_features=self.state_size,
            space=obs_space,
        )

        if cfg.reward_fn == "clip":
            rew_space = spaces.torch.Box((), *cfg.clip_rew)
        elif cfg.reward_fn == "sign":
            rew_space = spaces.torch.OneOf(torch.tensor([-1.0, 0.0, 1.0]))
        elif cfg.reward_fn == "tanh":
            rew_space = spaces.torch.Box((), -1.0, +1.0)
        else:
            rew_space = spaces.torch.Box((), -torch.inf, +torch.inf)

        self.reward_dec = self._make_decoder(cfg.decoders.reward)(
            in_features=self.state_size,
            space=rew_space,
        )

        self.term_dec = self._make_decoder(cfg.decoders.term)(
            in_features=self.state_size,
            space=spaces.torch.Discrete(2, dtype=torch.bool),
        )

        self.to(device)

        self.opt = self._make_optim(cfg.opt)(self.parameters())

    @cached_property
    def scaler(self):
        return getattr(torch, self.device.type).amp.GradScaler()

    def autocast(self):
        enabled = self.dtype != torch.float32
        return torch.autocast(self.device.type, self.dtype, enabled)

    def _make_encoder(self, cfg: dict) -> Callable[..., nn.Module]:
        cfg = {**cfg}
        cls = config.get_class(nets, cfg["$type"] + "_encoder")
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_decoder(self, cfg: dict) -> Callable[..., nn.Module]:
        cfg = {**cfg}
        cls = config.get_class(nets, cfg["$type"] + "_decoder")
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_optim(self, cfg: dict) -> Callable[..., torch.optim.Optimizer]:
        cfg = {**cfg}
        cls = config.get_class(torch.optim, cfg["$type"])
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_inv(self, encoder: nn.Module):
        if isinstance(encoder, nets.DiscreteEncoder):
            return encoder.inverse
        elif isinstance(encoder, nets.Identity):
            return encoder
        else:
            raise ValueError(f"Cannot create a decoder for {type(encoder)}")

    def opt_step(self, batch: dict, state: rssm.State):
        mets = {}
        losses = {}

        with self.autocast():
            enc_obs = over_seq(self.obs_enc)(batch["obs"])
            enc_act = over_seq(self.act_enc)(batch["act"]).type_as(enc_obs)
            enc_act[torch.where(batch["first"])].zero_()

            states, post, prior = self.rssm.observe(enc_obs, enc_act, state)
            mets["prior_ent"] = prior.detach().entropy().mean()
            mets["post_ent"] = post.detach().entropy().mean()
            kl_loss, kl_value = self._kl_loss(post, prior, **self.cfg.kl)
            losses["kl"] = kl_loss
            mets["kl_value"] = kl_value.detach().mean()

            feats = states.to_tensor()

            obs_dist = over_seq(self.obs_dec)(feats)
            obs = batch["obs"]
            losses["obs"] = -obs_dist.log_prob(obs).mean()

            rew_dist = over_seq(self.reward_dec)(feats)
            reward = self.reward_fn(batch["reward"])
            losses["reward"] = -rew_dist.log_prob(reward).mean()

            term_dist = over_seq(self.term_dec)(feats)
            term = batch["term"]
            losses["term"] = -term_dist.log_prob(term).mean()

            coef = self.cfg.coef
            loss: Tensor = sum(
                coef[k] * v if k in coef else v for k, v in losses.items()
            )

        mets.update(
            {
                "loss": loss.detach(),
                **{f"{k}_loss": v.detach() for k, v in losses.items()},
            }
        )

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if self.cfg.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.cfg.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()

        states = states.detach()

        return mets, states

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
