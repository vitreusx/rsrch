from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from dreamerx.common import plasticity
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import frozen, over_seq, safe_mode
from rsrch.rl.utils import polyak

from ..common import nets
from ..common.config import MakeSched
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import to_camel_case
from . import _alpha as alpha
from ._utils import gae_only_ret


@dataclass
class Config:
    @dataclass
    class Actor:
        encoder: dict
        dist: dict
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
    gae_lambda: float
    alpha: alpha.Config
    clip_grad: float | None
    rew_fn: Literal["id", "sign", "tanh"] = "id"


def layer_init(layer, bias_const=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ContQf(nn.Module):
    def __init__(
        self,
        cfg: Config.Qf,
        obs_space: spaces.torch.Box,
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

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        input = torch.cat((obs.flatten(1), act.flatten(1)), 1)
        q_value = self.proj(self.encoder(input))
        return q_value.ravel()


class DiscQf(nn.Module):
    def __init__(
        self,
        cfg: Config.Qf,
        obs_space: spaces.torch.Tensor,
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
                q_values = (q_values * act).sum(-1)
            else:
                q_values = q_values.gather(1, act.unsqueeze(-1)).squeeze(-1)
        return q_values


def Qf(cfg: Config.Qf, obs_space, act_space):
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
            **cfg.dist,
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
        make_sched: MakeSched | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor

        act_space = actor.act_space

        self.qf, self.qf_t = nn.ModuleList(), nn.ModuleList()
        for _ in range(cfg.num_qf):
            self.qf.append(self._make_qf())
            self.qf_t.append(self._make_qf())
        polyak.sync(self.qf, self.qf_t)

        self.actor_opt = self._make_opt(self.actor.parameters(), cfg.actor.opt)
        self.qf_opt = self._make_opt(self.qf.parameters(), cfg.qf.opt)
        self.opt_iter = 0

        self.qf_polyak = polyak.Polyak(self.qf, self.qf_t, **cfg.qf.polyak)
        self.alpha = alpha.Alpha(cfg.alpha, act_space, self.device, make_sched)

        self._discrete = isinstance(self.qf[0], DiscQf)

        self._actor_ref = plasticity.save_ref_state(self.actor)
        self._qf_ref = plasticity.save_ref_state(self.qf[0])

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
        batch = batch.detach()
        obs, next_obs = batch.obs[:-1], batch.obs[1:]

        with torch.no_grad():
            cont = 1.0 - batch.term.float()
            weight = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])
            weight = torch.cumprod(weight, 0)

        with self.autocast():
            policy = over_seq(self.actor)(batch.obs)

        with torch.no_grad():
            with self.autocast():
                next_policy = policy[1:].detach()

                if self._discrete:
                    min_q = over_seq(self.qf_t[0])(next_obs)
                    for idx in range(1, self.cfg.num_qf):
                        min_q_idx = over_seq(self.qf_t[idx])(next_obs)
                        min_q = torch.min(min_q, min_q_idx)
                    policy: D.Categorical | D.OneHot
                    q_values = min_q - self.alpha.value * next_policy.log_probs
                    next_v = (next_policy.probs * q_values).sum(-1)
                else:
                    next_act = next_policy.sample()
                    min_q = over_seq(self.qf_t[0])(next_obs, next_act)
                    for idx in range(1, self.cfg.num_qf):
                        min_q_idx = over_seq(self.qf_t[idx])(next_obs, next_act)
                        min_q = torch.min(min_q, min_q_idx)
                    next_v = min_q - self.alpha.value * next_policy.log_prob(next_act)

                gamma = cont * self.cfg.gamma
                reward = self._transform_reward(batch.reward)
                target = gae_only_ret(reward, next_v, gamma[1:], self.cfg.gae_lambda)

        with self.autocast():
            q_losses = []
            for qf in self.qf:
                qf_pred = over_seq(qf)(obs, batch.act)
                q_loss = (weight[:-1] * (qf_pred - target).square()).mean()
                q_losses.append(q_loss)
            q_loss = torch.stack(q_losses).sum()

        self.qf_opt.step(q_loss, self.cfg.clip_grad)
        self.qf_polyak.step()

        cur_policy = policy[:-1]
        if self._discrete:
            with torch.no_grad():
                with self.autocast():
                    min_q = over_seq(self.qf[0])(obs)
                    for idx in range(1, self.cfg.num_qf):
                        min_q_idx = over_seq(self.qf[idx])(obs)
                        min_q = torch.min(min_q, min_q_idx)

            with self.autocast():
                policy: D.Categorical
                actor_losses = self.alpha.value * cur_policy.log_probs - min_q
                actor_losses = (cur_policy.probs * actor_losses).sum(-1)
                actor_loss = (weight[:-1] * actor_losses).mean()
        else:
            with self.autocast():
                act = cur_policy.rsample()
                with frozen(self.qf):
                    min_q = over_seq(self.qf[0])(obs, act)
                    for idx in range(1, self.cfg.num_qf):
                        min_q_idx = over_seq(self.qf[idx])(obs, act)
                        min_q = torch.min(min_q, min_q_idx)
                actor_losses = self.alpha.value * cur_policy.log_prob(act) - min_q
                actor_loss = (weight[:-1] * actor_losses).mean()

        self.actor_opt.step(actor_loss, self.cfg.clip_grad)

        with torch.no_grad():
            entropy = policy.entropy()

        if self.alpha.adaptive:
            self.alpha.opt_step(entropy)

        self.opt_iter += 1

        with torch.no_grad():
            mets = {
                "q_loss": q_loss / self.cfg.num_qf,
                "mean_q": qf_pred.mean(),
                "actor_loss": actor_loss,
                "entropy": entropy.mean(),
                "alpha": self.alpha.value,
            }

        return mets

    def _transform_reward(self, reward: Tensor):
        if self.cfg.rew_fn == "tanh":
            return torch.tanh(reward)
        elif self.cfg.rew_fn == "sign":
            return torch.sign(reward)
        elif self.cfg.rew_fn == "id":
            return reward

    def check_plasticity(self, batch: Slices):
        obs = batch.obs.flatten(0, 1)

        _, actor_res = plasticity.full_test(
            module=self.actor,
            ref_state=self._actor_ref,
            input=(obs,),
        )
        actor_mets = plasticity.full_metrics(actor_res)

        _, qf_res = plasticity.full_test(
            module=self.qf[0],
            ref_state=self._qf_ref,
            input=(obs,),
        )
        qf_mets = plasticity.full_metrics(qf_res)

        metrics = {
            **{f"qf/{k}": v for k, v in actor_mets.items()},
            **{f"critic/{k}": v for k, v in qf_mets.items()},
        }
        results = {"actor": actor_res, "qf": qf_res}

        return metrics, results
