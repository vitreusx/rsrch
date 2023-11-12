from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.exp import Board
from rsrch.rl import data, gym
from rsrch.utils import cron

from ..common.utils import Optim, flat, over_seq


class RNN:
    num_layers: int
    """Number of layers in the RNN. Determines shape of hidden state."""

    def __call__(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Given prior state and a sequence of inputs, get (1) state at final
        time step, (2) final layer states for each element of the sequence.
        :param x: Input, of shape (L, N, D_x).
        :param h: State, of shape (#Layers, N, D_h).
        :return: Tuple (hx, out) with final layer states and final time step state,
        of shapes (L, N, D_h) and (#Layers, N, D_h) respectively.
        """


class WorldModel:
    """Hybrid world model. Combines RNN-based (deterministic) belief model with
    a state distribution prediction head on top.
    """

    def obs_enc(self, obs: Tensor) -> Tensor:
        """Given a batch of observations, encode them."""

    def act_enc(self, act: Tensor) -> Tensor:
        """Given a batch of actions, encode them."""

    def init(self, obs: Tensor) -> Tensor:
        """Given initial observation, get initial state for transition RNN."""

    trans: RNN
    """Transition RNN. Accepts concatenated actions and observations from next
    states as input. Returns (deterministic) belief states, which can be 
    projected to a latent state distribution."""

    def trans_proj(self, hx: Tensor) -> D.Distribution:
        """Given belief states from the transition RNN or prediction cell,
        get the distribution over latent states."""

    def pred(self, act: Tensor, s: Tensor) -> D.Distribution:
        """Prediction model. Given latent state and action, get a distribution
        of next state."""

    def recon(self, s: Tensor) -> D.Distribution:
        """Given latent states, reconstruct observations."""

    def reward(self, next_s: Tensor) -> D.Distribution:
        """Given latent states, get a distribution over rewards-upon-arriving."""

    def term(self, s: Tensor) -> D.Distribution:
        """Given latent states, get a distribution over being-terminal."""


class Actor:
    def __call__(self, hx: Tensor) -> D.Distribution:
        """For a batch of states, get a distribution over (encoded) actions
        to take."""

    def act_dec(self, enc_act: Tensor) -> Tensor:
        """For a batch of encoded actions, as sampled from the policy
        distribution, decode them and get 'true' actions."""


class VecAgent(gym.vector.Agent):
    def __init__(self, wm: WorldModel, actor: Actor):
        super().__init__()
        self.wm = wm
        self.actor = actor
        self._h = None

    def reset(self, idxes, obs, info):
        obs = self.wm.obs_enc(obs)
        if self._h is None:
            self._h = self.wm.init(obs)
        else:
            self._h[:, idxes] = self.wm.init(obs)

    def policy(self, obs):
        state = self.wm.trans_proj(self._h[-1]).sample()
        return self.actor.act_dec(self.actor(state).sample())

    def step(self, act):
        act = self.wm.act_enc(act)
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.wm.obs_enc(next_obs)
        x = torch.cat([self._act[idxes], next_obs], -1)[None]
        h0 = self._h[:, idxes]
        _, next_h = self.wm.trans(x, h0)
        self._h[:, idxes] = next_h


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
    board: Board


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

    def _mixed_kl(self, post: D.Distribution, prior: D.Distribution):
        to_post = D.kl_divergence(post.detach(), prior)
        to_prior = D.kl_divergence(post, prior.detach())
        return self.cfg.kl_mix * to_post + (1.0 - self.cfg.kl_mix) * to_prior

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
            kl_divs = self._mixed_kl(flat(trans_rv[1:]), flat(pred_rv))
            pred_loss = c.pred * kl_divs.mean()
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
            board = self.ctx.board
            for k in ["obs", "rew", "term", "wm", "pred"]:
                if f"{k}_loss" not in locals():
                    continue
                board.add_scalar(f"train/{k}_loss", locals()[f"{k}_loss"])

        # Return the states from transition RNN for further processing.
        return hx.detach()
