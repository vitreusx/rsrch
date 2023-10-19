from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn.builder import *
from rsrch.rl import gym
from rsrch.rl.utils.polyak import Polyak
from ..proto import *


@dataclass
class Config:
    @dataclass
    class Polyak:
        enabled: bool
        period: int
        tau: float

    @dataclass
    class Alpha:
        autotune: bool
        ent_scale: float | None
        value: float | None
        opt: optim_ctor

    actor_opt: optim_ctor
    critic_opt: optim_ctor
    target_v: Polyak
    min_v: bool
    adv_norm: bool
    alpha: Alpha
    horizon: int
    rho: float
    gamma: float
    gae_lambda: float


def max_ent(space: gym.TensorSpace):
    """Get maximum entropy of a policy over a given action space."""
    if isinstance(space, gym.spaces.TensorDiscrete):
        return np.log(space.n)
    elif isinstance(space, gym.spaces.TensorBox):
        return torch.log(space.high - space.low).sum()


def flat(x: Tensor):
    return x.flatten(0, 1)


class Trainer(nn.Module):
    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        critic_fn: Callable[[], Critic],
        wm: WorldModel,
        act_space: gym.TensorSpace,
    ):
        super().__init__()
        self.cfg = cfg
        self._wm = wm

        self.pi = actor
        self.pi_opt = cfg.actor_opt(self.pi.parameters())

        self.v1 = critic_fn()
        if cfg.min_v:
            self.v2 = critic_fn()
            v_params = [*self.v1.parameters(), *self.v2.parameters()]
            self.v_opt = cfg.critic_opt(v_params)
        else:
            self.v2 = None
            self.v_opt = cfg.critic_opt(self.v1.parameters())

        if cfg.target_v.enabled:
            self.v1_t = deepcopy(self.v1)
            self.v1_polyak = Polyak(
                source=self.v1,
                target=self.v1_t,
                tau=cfg.target_v.tau,
                every=cfg.target_v.period,
            )
            if self.v2 is not None:
                self.v2_t = deepcopy(self.v2)
                self.v2_polyak = Polyak(
                    source=self.v2,
                    target=self.v2_t,
                    tau=cfg.target_v.tau,
                    every=cfg.target_v.period,
                )

        if cfg.alpha.autotune:
            self._target_ent = cfg.alpha.ent_scale * max_ent(act_space)
            self.log_alpha = nn.Parameter(torch.zeros([]), requires_grad=True)
            self.alpha_opt = cfg.alpha.opt([self.log_alpha])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = cfg.alpha.value

    def opt_step(self, init: Tensor, ctx):
        states, act_rvs, acts = [init], [], []
        for _ in range(self.cfg.horizon):
            act_rv = self.pi(states[-1].detach())
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            next_s = self._wm.step(states[-1], act).rsample()
            states.append(next_s)

        states = torch.stack(states)
        acts = torch.stack(acts)
        act_rvs: D.Distribution = torch.stack(act_rvs)

        seq_len, batch_size = states.shape[:2]

        if self.cfg.min_v:
            v1 = self.v1(flat(states)).mean
            v1 = v1.reshape(seq_len, batch_size, *v1.shape[1:])
            v2 = self.v2(flat(states)).mean
            v2 = v2.reshape(seq_len, batch_size, *v2.shape[1:])
            v = torch.minimum(v1, v2)
        else:
            v = self.v1(flat(states)).mean
            v = v.reshape(seq_len, batch_size, *v.shape[1:])

        if self.cfg.target_v.enabled:
            with torch.no_grad():
                if self.cfg.min_v:
                    v1_t = self.v1_t(flat(states)).mean
                    v2_t = self.v2_t(flat(states)).mean
                    vt = torch.minimum(v1_t, v2_t)
                else:
                    vt = self.v1_t(flat(states)).mean
                vt = vt.reshape(seq_len, batch_size, *vt.shape[1:])
        else:
            vt = v.detach()

        with torch.no_grad():
            term = self._wm.term(flat(states)).mean.float()
            term = term.reshape(seq_len, batch_size, *term.shape[1:])
            term[0] = 0
            rew = self._wm.reward(flat(states[1:])).mean
            rew = rew.reshape(seq_len - 1, batch_size, *rew.shape[1:])

        w = torch.cumprod(1.0 - term, 0)
        gamma = self.cfg.gamma * w
        with torch.no_grad():
            delta = (rew + gamma[1:] * vt[1:]) - vt[:-1]
            adv = [delta[-1]]
            for t in reversed(range(1, len(rew))):
                adv.append(delta[t - 1] + self.cfg.gae_lambda * gamma[t] * adv[-1])
            adv.reverse()
            adv = torch.stack(adv)
            ret = vt[:-1] + adv

        if self.cfg.adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        w = w[:-1]

        if self.cfg.min_v:
            v1_loss = (v1[:-1] - ret).square()
            v2_loss = (v2[:-1] - ret).square()
            critic_loss = v1_loss + v2_loss
        else:
            critic_loss = (v[:-1] - ret).square()

        self.v_opt.zero_grad(set_to_none=True)
        (w * critic_loss).mean().backward(retain_graph=True)
        self.v_opt.step()
        if self.cfg.target_v.enabled:
            self.v1_polyak.step()
            if self.cfg.min_v:
                self.v2_polyak.step()

        logp = act_rvs.log_prob(acts)
        ent = act_rvs.entropy()
        pg_loss = -adv * logp
        actor_v_loss = -v[:-1]
        ent_loss = self.alpha * -ent
        pg_coef, v_coef = self.cfg.rho, 1.0 - self.cfg.rho
        actor_loss = pg_coef * pg_loss + v_coef * actor_v_loss + ent_loss

        self.pi_opt.zero_grad(set_to_none=True)
        (w * actor_loss).mean().backward()
        self.pi_opt.step()

        if self.cfg.alpha.autotune:
            alpha_loss = self.log_alpha * (ent - self._target_ent).detach()
            self.alpha_opt.zero_grad(set_to_none=True)
            (w * alpha_loss).mean().backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        if ctx.should_log:
            w_avg = lambda x: (w * x).mean() / w.mean()
            ctx.board.add_scalar(f"train/policy_ent", w_avg(ent))
            for k in ["critic", "pg", "actor_v", "ent", "actor"]:
                value = locals()[f"{k}_loss"]
                ctx.board.add_scalar(f"train/{k}_loss", w_avg(value))

            if self.cfg.alpha.autotune:
                ctx.board.add_scalar("train/alpha", self.alpha)
                ctx.board.add_scalar("train/alpha_loss", w_avg(alpha_loss))
