from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

import rsrch.distributions as D
from rsrch.exp.api import Experiment
from rsrch.rl import data
from rsrch.utils import cron

from ..common.utils import Optim, flat, over_seq
from .wm import WorldModel


@dataclass
class Config:
    @dataclass
    class Coefs:
        pred: float
        obs: float
        rew: float
        term: float

    batch_size: int
    seq_len: int
    opt: Optim
    coefs: Coefs
    kl_mix: float


class Context:
    should_log: cron.Flag
    exp: Experiment


class Trainer:
    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        ctx: Context,
    ):
        self.cfg = cfg
        self.ctx = ctx
        self.wm = wm
        self.wm_opt = cfg.opt.make()(self.wm.parameters())

    def _dist_loss(self, post: D.Distribution, prior: D.Distribution):
        to_prior = D.kl_divergence(post, prior.detach())
        to_post = D.kl_divergence(post.detach(), prior)
        return self.cfg.kl_mix * to_prior + (1.0 - self.cfg.kl_mix) * to_post

    def _data_loss(self, dist: D.Distribution, value: Tensor, reduction="mean"):
        if isinstance(dist, D.Dirac):
            return F.mse_loss(value, dist.value, reduction=reduction)
        else:
            loss = -dist.log_prob(value)
            if reduction == "mean":
                loss = loss.mean()
            return loss

    def opt_step(self, batch: data.ChunkBatch):
        """Take an optimization step, using a batch of env rollouts.
        Returns WM states for each observation"""

        self.wm.train()

        # Encode observations and actions
        obs = over_seq(self.wm.obs_enc)(batch.obs)  # [L+1, N, D_obs]
        act = over_seq(self.wm.act_enc)(batch.act)  # [L, N, D_act]

        # Get posterior states using transition RNN
        h0 = self.wm.init(obs[0])  # [#Layers, N, D_h]
        trans_x = torch.cat([act, obs[1:]], -1)  # [L, N, D_obs + D_act]
        hx, _ = self.wm.trans(trans_x, h0.contiguous())  # [L, N, D_h]
        hx = torch.cat([h0[-1][None], hx], 0)  # [L+1, N, D_h]

        c = self.cfg.coefs
        losses = []

        # Compute prediction loss as the KL divergence.
        trans_rv = over_seq(self.wm.trans_proj)(hx)
        if c.pred != 0:
            pred_rv = over_seq(self.wm.pred)(hx[:-1], act)
            divs = self._dist_loss(flat(trans_rv[1:]), flat(pred_rv))
            pred_loss = c.pred * divs.mean()
            losses.append(pred_loss)

        # Sample transition-RNN state.
        states = trans_rv.rsample()

        # Compute reconstruction/modelling losses.
        if c.obs != 0:
            obs_rv = self.wm.recon(flat(states))
            obs_loss = c.obs * self._data_loss(obs_rv, flat(batch.obs))
            losses.append(obs_loss)

        if c.rew != 0:
            rew_rv = self.wm.reward(flat(states[1:]))
            rew_loss = c.rew * self._data_loss(rew_rv, flat(batch.reward))
            losses.append(rew_loss)

        if c.term != 0:
            term_rv = self.wm.term(states[-1])
            term_loss = c.term * self._data_loss(term_rv, batch.term.float())
            losses.append(term_loss)

        # Take optimization step.
        wm_loss = sum(losses)
        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        self.wm_opt.step()

        if self.ctx.should_log:
            exp = self.ctx.exp
            for k in ["obs", "rew", "term", "wm", "pred"]:
                if f"{k}_loss" not in locals():
                    continue
                exp.add_scalar(f"train/{k}_loss", locals()[f"{k}_loss"])

        # Return the states from transition RNN for further processing.
        return hx.detach()
