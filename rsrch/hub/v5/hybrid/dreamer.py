from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

import rsrch.distributions as D
from rsrch.exp import Board
from rsrch.rl import data, gym
from rsrch.rl.utils import polyak
from rsrch.utils import cron

from ..common.utils import Optim, Polyak, over_seq
from .wm import WorldModel


class Actor:
    def __call__(self, state: Tensor) -> D.Distribution:
        """Get a distribution over actions for a batch of states. The distribution
        is over the encoded actions, to make function differentiable when the
        action space is discrete.
        """


class Critic:
    def __call__(self, state: Tensor) -> Tensor:
        """Get state values for a batch of states."""


@torch.compile
def gae_adv_est(
    rew: Tensor,
    val: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Perform Generalized Advantage Estimation (GAE).
    :param rew: Tensor of shape (L, N) containing rewards-upon-reaching for states.
    To be specific, rew[t] denotes batch of rewards upon reaching state s[t+1].
    :param val: Tensor of shape (L+1, N) containing values at given states. For
    terminal or post-terminal time-steps, values should be equal to zero.
    :return: A tuple of advantage estimates and cumulative returns, of shapes
    (L, N) each.
    """

    delta = (rew + gamma * val[1:]) - val[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, rew.shape[0])):
        adv_t = delta[t - 1] + gamma * gae_lambda * adv[-1]
        adv.append(adv_t)
    adv.reverse()
    adv = torch.stack(adv)
    ret = val[:-1] + adv
    return adv, ret


def max_ent(space: gym.TensorSpace):
    if isinstance(space, gym.spaces.TensorDiscrete):
        return np.log(space.n)
    elif isinstance(space, gym.spaces.TensorBox):
        return torch.log(space.high - space.low).sum()


@dataclass
class Config:
    @dataclass
    class Alpha:
        autotune: bool
        ent_scale: float | None
        value: float | None
        opt: Optim

    @dataclass
    class Coefs:
        critic: float
        actor_pg: float
        actor_v: float

    batch_size: int
    horizon: int
    gamma: float
    gae_lambda: float
    adv_norm: bool
    alpha: Alpha
    polyak: Polyak
    coefs: Coefs
    opt: Optim


OptimizerF = Callable[[list[nn.Parameter]], Optimizer]


class Context:
    should_log: cron.Flag
    board: Board


class Trainer:
    """Model-based AC trainer."""

    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        act_space: gym.TensorSpace,
        actor: Actor,
        critic_ctor: Callable[[], Critic],
        ctx: Context,
    ):
        self.ctx = ctx
        self.cfg = cfg
        self.wm = wm

        self.actor = actor
        self.v = critic_ctor()
        ac_params = [*self.actor.parameters(), *self.v.parameters()]
        self.ac_opt = cfg.opt.make()(ac_params)

        self.vt = critic_ctor()
        polyak.sync(self.v, self.vt)
        self.v_polyak = cfg.polyak.make()(self.v, self.vt)

        if cfg.alpha.autotune:
            self._target_ent = cfg.alpha.ent_scale * max_ent(act_space)
            self.log_alpha = nn.Parameter(torch.zeros([]), requires_grad=True)
            self.alpha_opt = cfg.alpha.opt.make()([self.log_alpha])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.alpha.value

    def opt_step(self, init_s: Tensor):
        """Take an opt step via training actor-critic setup in latent
        imagination, starting from given states."""

        cur_s, all_s = init_s, [init_s]
        act_rvs, acts = [], []
        for _ in range(self.cfg.horizon):
            act_rv = self.actor(cur_s)
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            next_s = self.wm.pred(act, cur_s).rsample()
            all_s.append(next_s)
            cur_s = next_s

        all_s = torch.stack(all_s)  # [L+1, N, H]
        acts = torch.stack(acts)  # [L, N, H]
        act_rvs: D.Distribution = torch.stack(act_rvs)  # [L, H]

        with torch.no_grad():
            rew_rvs: D.Distribution = over_seq(self.wm.reward)(all_s[1:])  # [L, H]
            rew = rew_rvs.mean
            term_rvs: D.Distribution = over_seq(self.wm.term)(all_s)  # [L+1, H]
            term = term_rvs.mode
            cont = loss_w = (1.0 - term).cumprod(0)

        val = over_seq(self.v)(all_s)
        val = val * cont
        logp = act_rvs.log_prob(acts)
        ent = act_rvs.entropy()

        adv, ret = gae_adv_est(rew, val, self.cfg.gamma, self.cfg.gae_lambda)
        if self.cfg.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        coefs = self.cfg.coefs
        losses = []
        if coefs.critic != 0:
            critic_loss = coefs.critic * (val[:-1] - ret).square()
            losses.append(critic_loss)
        if coefs.actor_pg != 0:
            actor_pg_loss = coefs.actor_pg * -adv * logp
            losses.append(actor_pg_loss)
        if coefs.actor_v != 0:
            actor_v_loss = coefs.actor_v * -val[:-1]
            losses.append(actor_v_loss)
        actor_ent_loss = self.alpha * -ent
        losses.append(actor_ent_loss)

        ac_loss = sum(losses)

        self.ac_opt.zero_grad(set_to_none=True)
        (ac_loss * loss_w[:-1]).mean().backward()
        self.ac_opt.step()

        if self.cfg.alpha.autotune:
            alpha_loss = self.log_alpha * (ent - self._target_ent).detach()
            self.alpha_opt.zero_grad(set_to_none=True)
            (alpha_loss * loss_w[:-1]).mean().backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # self.v_polyak.step()

        if self.ctx.should_log:
            avg = lambda x: (loss_w[:-1] * x).mean() / (loss_w[:-1].mean() + 1e-8)
            board = self.ctx.board
            for k in ["critic", "actor_pg", "actor_v", "actor_ent", "ac"]:
                if f"{k}_loss" not in locals():
                    continue
                loss_v = locals()[f"{k}_loss"]
                board.add_scalar(f"train/{k}_loss", avg(loss_v))

            if self.cfg.alpha.autotune:
                board.add_scalar("train/alpha", self.alpha)
                board.add_scalar("train/alpha_loss", avg(alpha_loss))
