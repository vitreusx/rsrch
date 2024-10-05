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
from rsrch.nn.utils import frozen, safe_mode
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
        encoder: dict

    @dataclass
    class Q:
        encoder: dict
        polyak: dict

    opt: dict
    actor: Actor
    qf: Q
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
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor

        act_space = actor.act_space
        self._discrete = type(act_space) == spaces.torch.Discrete

        self.qf, self.qf_t = nn.ModuleList(), nn.ModuleList()
        for _ in range(cfg.num_q):
            self.qf.append(self._make_q())
            self.qf_t.append(self._make_q())
        polyak.sync(self.qf, self.qf_t)

        self._opt_groups = {
            "actor": [*self.actor.parameters()],
            "qf": [*self.qf.parameters()],
        }
        self.opt = self._make_opt(self._opt_groups, cfg.opt)
        self.opt_iter = 0

        self.qf_polyak = polyak.Polyak(self.qf, self.qf_t, **cfg.qf.polyak)
        self.alpha = alpha.Alpha(cfg.alpha, act_space)

        self._save_init_parameters()

    def _save_init_parameters(self):
        self._init_params = {}
        for name, params in self._opt_groups.items():
            self._init_params[name] = [p.data.clone() for p in params]

    def _make_q(self):
        q = Q(self.cfg.qf, self.actor.obs_space, self.actor.act_space)
        q = q.to(self.device)
        return q

    def _make_opt(self, groups: dict[str, nn.Parameter], cfg: dict):
        cfg = {**cfg}
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]

        opt_groups = []
        for name, params in [*groups.items()]:
            if name in cfg:
                opt_groups.append({"params": params, **cfg[name]})
                del cfg[name]
            else:
                opt_groups.append({"params": params})

        opt = cls(opt_groups, **cfg)
        opt = ScaledOptimizer(opt)
        return opt

    def opt_step(self, batch: Slices):
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

            policy = self.actor(obs)
            act = policy.rsample()

        if self._discrete:
            with torch.no_grad():
                with self.autocast():
                    min_q = self.qf[0](obs)
                    for idx in range(1, self.cfg.num_q):
                        min_q = torch.min(min_q, self.qf[idx](obs))

            with self.autocast():
                policy: D.Categorical
                actor_losses = -(min_q - self.alpha.value * policy.log_probs)
                actor_loss = (policy.probs * actor_losses).mean()
                actor_loss = actor_loss * len(obs)
        else:
            with self.autocast():
                with frozen(self.qf):
                    min_q = self.qf[0](obs, act)
                    for idx in range(1, self.cfg.num_q):
                        min_q = torch.min(min_q, self.qf[idx](obs, act))
                actor_loss = -(min_q - self.alpha.value * policy.log_prob(act)).mean()

        loss = q_loss + actor_loss

        with torch.no_grad():
            mets = {
                "q_loss": q_loss,
                "mean_q": qf_pred.mean(),
                "actor_loss": actor_loss,
            }

        self.opt.step(loss, self.cfg.clip_grad)
        self.qf_polyak.step()
        self.opt_iter += 1

        if self.alpha.adaptive:
            entropy = self.actor(batch.obs[0]).entropy()
            alpha_mets = self.alpha.opt_step(entropy)

            with torch.no_grad():
                if self.cfg.alpha.adaptive:
                    mets.update({f"alpha/{k}": v for k, v in alpha_mets.items()})

        return mets

    @torch.no_grad()
    def reset(self, make_actor: Callable[[], Actor]):
        new_actor = make_actor()
        polyak.sync(new_actor, self.actor)

        for idx in range(self.cfg.num_q):
            new_qf = self._make_q()
            polyak.sync(new_qf, self.qf[idx])
            polyak.sync(new_qf, self.qf_t[idx])

        self._save_init_parameters()

    @torch.no_grad()
    def compute_stats(self):
        for name in self._init_params:
            init_values = self._init_params[name]
            cur_values = self._opt_groups[name]

            p_norms, rel_p_norms, dp_norms, rel_dp_norms = [], [], [], []
            for init_p, cur_p in zip(init_values, cur_values):
                p_norm = torch.linalg.norm(cur_p.data)
                p_norms.append(p_norm)
                dp_norm = torch.linalg.norm(cur_p.data - init_p)
                dp_norms.append(dp_norm)
                p0_norm = torch.linalg.norm(init_p)
                if p0_norm != 0.0:
                    rel_p_norms.append(p_norm / p0_norm)
                    rel_dp_norms.append(dp_norm / p0_norm)

            stats = {}
            stats[f"{name}/p_norm"] = torch.mean(torch.stack(p_norms))
            stats[f"{name}/dp_norm"] = torch.mean(torch.stack(dp_norms))
            if len(rel_p_norms) > 0:
                stats[f"{name}/rel_p_norm"] = torch.mean(torch.stack(rel_p_norms))
                stats[f"{name}/rel_dp_norm"] = torch.mean(torch.stack(rel_dp_norms))

        return stats
