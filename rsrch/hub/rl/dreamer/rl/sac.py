from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import frozen, safe_mode
from rsrch.rl.utils import polyak
from rsrch.types import Tensorlike

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
        opt: dict

    @dataclass
    class Qf:
        encoder: dict
        polyak: dict
        opt: dict

    actor: Actor
    qf: Qf
    num_qf: int
    gamma: float
    alpha: alpha.Config
    clip_grad: float | None
    rew_fn: Literal["id", "clip", "tanh"] = "id"
    rew_clip: tuple[float, float] | None = None


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ContQf(nn.Module):
    def __init__(
        self,
        cfg: Config.Qf,
        obs_space: spaces.torch.Box | spaces.torch.Tensorlike,
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

        self.proj = layer_init(nn.Linear(z_features, 1))

    def forward(self, obs: Tensor | Tensorlike, act: Tensor) -> Tensor:
        if not isinstance(obs, Tensor):
            obs = obs.as_tensor()
        input = torch.cat((obs.flatten(1), act.flatten(1)), 1)
        q_value = self.proj(self.encoder(input))
        return q_value.ravel()


class DiscQf(nn.Module):
    def __init__(
        self,
        cfg: Config.Qf,
        obs_space: spaces.torch.Tensor | spaces.torch.Tensorlike,
        act_space: spaces.torch.Discrete | spaces.torch.OneHot,
    ):
        super().__init__()
        self.act_space = act_space

        self.encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = obs_space.sample((1,))
            z_features = self.encoder(input).shape[1]

        self.proj = layer_init(nn.Linear(z_features, act_space.n))

    def forward(self, obs: Tensor, act: Tensor | None = None) -> Tensor:
        q_values = self.proj(self.encoder(obs))
        if act is not None:
            if act.dtype.is_floating_point:
                # One-hot encoded actions
                q_values = (q_values * act).sum(-1)
            else:
                # Discrete actions
                q_values = q_values.gather(1, act.unsqueeze(-1)).squeeze(-1)
        return q_values


def Qf(
    cfg: Config.Qf,
    obs_space: spaces.torch.Tensor,
    act_space: spaces.torch.Tensor,
):
    if isinstance(act_space, (spaces.torch.Discrete, spaces.torch.OneHot)):
        return DiscQf(cfg, obs_space, act_space)
    else:
        return ContQf(cfg, obs_space, act_space)


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
        for _ in range(cfg.num_qf):
            self.qf.append(self._make_qf())
            self.qf_t.append(self._make_qf())
        polyak.sync(self.qf, self.qf_t)

        self.actor_opt = self._make_opt(self.actor.parameters(), cfg.actor.opt)
        self.qf_opt = self._make_opt(self.qf.parameters(), cfg.qf.opt)
        self.opt_iter = 0

        self.qf_polyak = polyak.Polyak(self.qf, self.qf_t, **cfg.qf.polyak)
        self.alpha = alpha.Alpha(cfg.alpha, act_space, self.device)

    def _make_qf(self):
        qf = Qf(self.cfg.qf, self.actor.obs_space, self.actor.act_space)
        qf = qf.to(self.device)
        return qf

    def _make_opt(self, parameters, cfg: dict):
        cfg = {**cfg}
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]
        opt = cls(parameters, **cfg)
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
                    for idx in range(1, self.cfg.num_qf):
                        min_q = torch.min(min_q, self.qf_t[idx](next_obs))
                    next_policy: D.Categorical
                    targets = min_q - self.alpha.value * next_policy.log_probs
                    target = (next_policy.probs * targets).sum(-1)
                else:
                    min_q = self.qf_t[0](next_obs, next_act)
                    for idx in range(1, self.cfg.num_qf):
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

        with self.autocast():
            policy = self.actor(obs)

        if self._discrete:
            with torch.no_grad():
                with self.autocast():
                    min_q = self.qf[0](obs)
                    for idx in range(1, self.cfg.num_qf):
                        min_q = torch.min(min_q, self.qf[idx](obs))

            with self.autocast():
                policy: D.Categorical
                actor_losses = self.alpha.value * policy.log_probs - min_q
                actor_loss = (policy.probs * actor_losses).mean()
                actor_loss = actor_loss * actor_losses.shape[-1]
        else:
            with self.autocast():
                act = policy.rsample()
                min_q = self.qf[0](obs, act)
                for idx in range(1, self.cfg.num_qf):
                    min_q = torch.min(min_q, self.qf[idx](obs, act))
                actor_loss = (self.alpha.value * policy.log_prob(act) - min_q).mean()

        self.actor_opt.step(actor_loss, self.cfg.clip_grad)

        self.opt_iter += 1

        with torch.no_grad():
            entropy = self.actor(batch.obs[0]).entropy()

        if self.alpha.adaptive:
            self.alpha.opt_step(entropy)

        with torch.no_grad():
            mets = {
                "q_loss": q_loss,
                "mean_q": qf_pred.mean(),
                "actor_loss": actor_loss,
                "entropy": entropy.mean(),
                "alpha": self.alpha.value,
            }

        return mets

    @torch.no_grad()
    def reset(self, make_actor: Callable[[], Actor]):
        new_actor = make_actor()
        polyak.sync(new_actor, self.actor)

        for idx in range(self.cfg.num_qf):
            new_qf = self._make_qf()
            polyak.sync(new_qf, self.qf[idx])
            polyak.sync(new_qf, self.qf_t[idx])

        # self._save_ref_state()
