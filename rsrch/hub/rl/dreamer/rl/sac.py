from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import safe_mode
from rsrch.rl.utils import polyak

from ..common import nets
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import to_camel_case
from . import _alpha as alpha


@dataclass
class Config:
    @dataclass
    class Actor:
        opt: dict
        opt_ratio: int
        encoder: dict

    @dataclass
    class Q:
        opt: dict
        encoder: dict
        polyak: dict

    actor: Actor
    critic: Q
    num_q: int
    gamma: float
    alpha: alpha.Config
    clip_grad: float | None


class ContQ(nn.Module):
    def __init__(
        self,
        cfg: Config.Q,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Box,
    ):
        super().__init__()

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))
        input_space = spaces.torch.Tensor((obs_dim + act_dim,))

        self.encoder = nets.make_encoder(input_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = input_space.sample((1,))
            z_features = self.encoder(input).shape[1]

        self.proj = nn.Linear(z_features, 1)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        input = torch.cat((obs.flatten(1), act.flatten(1)), 1)
        q_value = self.proj(self.encoder(input))
        return q_value.ravel()


class DiscQ(nn.Module):
    def __init__(
        self,
        cfg: Config.Q,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()

        self.encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = obs_space.sample((1,))
            z_features = self.encoder(input).shape[1]

        self.proj = nn.Linear(z_features, act_space.n)

    def forward(self, obs: Tensor, act: Tensor | None = None) -> Tensor:
        q_values = self.proj(self.encoder(obs))
        if act is not None:
            q_values = q_values.gather(1, act.unsqueeze(-1)).squeeze(-1)
        return q_values


def Q(
    cfg: Config.Q,
    obs_space: spaces.torch.Tensor,
    act_space: spaces.torch.Tensor,
):
    if isinstance(act_space, spaces.torch.Discrete):
        return DiscQ(cfg, obs_space, act_space)
    else:
        return ContQ(cfg, obs_space, act_space)


class Actor(nn.Sequential):
    def __init__(
        self,
        cfg: Config.Actor,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(encoder):
            input = obs_space.sample((1,))
            z_features = encoder(input).shape[1]

        head = dh.make(
            layer_ctor=partial(nn.Linear, z_features),
            space=act_space,
        )

        super().__init__(encoder, head)
        self.obs_space = obs_space
        self.act_space = act_space


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        make_q: Callable[[], DiscQ | ContQ],
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor
        act_space = actor.act_space
        self._discrete = type(act_space) == spaces.torch.Discrete

        self.qf, self.qf_t = nn.ModuleList(), nn.ModuleList()
        for _ in range(cfg.num_q):
            self.qf.append(make_q())
            self.qf_t.append(make_q())
        polyak.sync(self.qf, self.qf_t)

        self.actor_opt = self._make_opt(self.actor.parameters(), cfg.actor.opt)
        self.qf_opt = self._make_opt(self.qf.parameters(), cfg.critic.opt)
        self.qf_polyak = polyak.Polyak(self.qf, self.qf_t, **cfg.critic.polyak)
        self.alpha = alpha.Alpha(cfg.alpha, act_space)

        self.opt_iter = 0

    def _make_opt(self, parameters: list[nn.Parameter], cfg: dict):
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]
        opt = cls(parameters, **cfg)
        opt = ScaledOptimizer(opt)
        return opt

    def opt_step(self, batch: Slices):
        mets = {}

        with torch.no_grad():
            with self.autocast():
                next_obs = batch.obs[-1]
                next_policy = self.actor(next_obs)
                next_act = next_policy.sample()

                if self._discrete:
                    min_q = self.qf_t[0](next_obs)
                    for idx in range(1, self.cfg.num_q):
                        min_q = torch.min(min_q, self.qf_t[idx](next_obs))
                    next_policy: D.Categorical
                    targets = min_q - self.alpha.value * next_policy.log_probs
                    target = (next_policy.probs * targets).sum(-1)
                else:
                    min_q = self.qf_t[0](next_obs, next_act)
                    for idx in range(1, self.cfg.num_q):
                        min_q = torch.min(min_q, self.qf_t[idx](next_obs, next_act))
                    target = min_q - self.alpha.value * next_policy.log_prob(next_act)

                target *= 1.0 - batch.term[-1].float()

                for reward_t in reversed(batch.reward.unbind()):
                    target = reward_t + self.cfg.gamma * target

        obs = batch.obs[0]

        with self.autocast():
            q_losses = []
            for qf in self.qf:
                qf_pred = qf(obs, batch.act[0])
                q_losses.append(F.mse_loss(qf_pred, target))
            q_loss = torch.stack(q_losses).sum()

        self.qf_opt.step(q_loss, self.cfg.clip_grad)
        self.qf_polyak.step()

        with torch.no_grad():
            mets.update({"q_loss": q_loss, "mean_q": qf_pred.mean()})

        if self.opt_iter % self.cfg.actor.opt_ratio == 0:
            for _ in range(self.cfg.actor.opt_ratio):
                with self.autocast():
                    policy = self.actor(obs)
                    act = policy.rsample()

                    if self._discrete:
                        min_q = self.qf[0](obs)
                        for idx in range(1, self.cfg.num_q):
                            min_q = torch.min(min_q, self.qf[idx](obs))
                        policy: D.Categorical
                        actor_losses = -(min_q - self.alpha.value * policy.log_probs)
                        actor_loss = (policy.probs * actor_losses).mean()
                        actor_loss = actor_loss * len(obs)
                    else:
                        min_q = self.qf[0](obs, act)
                        for idx in range(1, self.cfg.num_q):
                            min_q = torch.min(min_q, self.qf[idx](obs, act))
                        actor_loss = -(
                            min_q - self.alpha.value * policy.log_prob(act)
                        ).mean()

                self.actor_opt.step(actor_loss, self.cfg.clip_grad)

                if self.alpha.adaptive:
                    entropy = self.actor(obs).entropy()
                    alpha_mets = self.alpha.opt_step(entropy)

            with torch.no_grad():
                mets.update({"actor_loss": actor_loss})
                if self.cfg.alpha.adaptive:
                    mets.update({f"alpha/{k}": v for k, v in alpha_mets.items()})

        self.opt_iter += 1
        return mets
