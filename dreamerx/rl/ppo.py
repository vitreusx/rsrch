from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import over_seq, safe_mode

from ..common import nets
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import find_class
from . import _alpha as alpha
from ._utils import gen_adv_est


@dataclass
class Config:
    encoder: dict
    actor_dist: dict
    update_epochs: int
    update_batch: int
    adv_norm: bool
    clip_coef: float
    clip_vloss: bool
    gamma: float
    gae_lambda: float
    opt: dict
    clip_grad: float | None
    alpha: alpha.Config
    vf_coef: float
    share_encoder: bool
    rew_fn: Literal["id", "sign", "tanh"]


class Actor(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space

        self.encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = obs_space.sample((1,))
            self.z_features = self.encoder(input).shape[1]

        layer_ctor = partial(nn.Linear, self.z_features)
        self.head = dh.make(layer_ctor, act_space, **cfg.actor_dist)

    def forward(self, state: Tensor) -> D.Distribution:
        return self.head(self.encoder(state))

    def forward_features(self, state: Tensor):
        features = self.encoder(state)
        return self.head(features), features


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CriticHead(nn.Sequential):
    def __init__(self, in_features: int):
        super().__init__(
            layer_init(nn.Linear(in_features, 1), std=1.0),
            nn.Flatten(0),
        )


class Critic(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: spaces.torch.Tensor,
    ):
        super().__init__()

        self.encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = obs_space.sample((1,))
            z_features = self.encoder(input).shape[1]

        self.head = CriticHead(z_features)

    def forward(self, state: Tensor):
        return self.head(self.encoder(state))


class Data(NamedTuple):
    obs: Tensor
    act: Tensor
    logp: Tensor
    adv: Tensor
    ret: Tensor
    val: Tensor
    weight: Tensor


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(compute_dtype=compute_dtype)
        self.cfg = cfg
        self.actor = actor
        device = next(actor.parameters()).device
        if self.cfg.share_encoder:
            self.critic_head = CriticHead(actor.z_features).to(device)
            parameters = [*self.actor.parameters(), *self.critic_head.parameters()]
        else:
            self.critic = Critic(cfg, actor.obs_space).to(device)
            parameters = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = self._make_opt(parameters)
        self.alpha = alpha.Alpha(cfg.alpha, actor.act_space, device)

    def _make_opt(self, parameters):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        opt = cls(parameters, **cfg)
        return ScaledOptimizer(opt)

    def _forward_ac(self, obs: Tensor):
        if self.cfg.share_encoder:
            policy, features = self.actor.forward_features(obs)
            val = self.critic_head(features)
        else:
            policy = self.actor(obs)
            val = self.critic(obs)
        return policy, val

    def opt_step(self, batch: Slices | list[Slices]):
        with torch.no_grad():
            with self.autocast():
                if isinstance(batch, Slices):
                    data = self._process_data_equal_size(batch)
                else:
                    data = self._process_data_var_size(batch)

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(len(data.val))
            for idxes in perm.split(self.cfg.update_batch):
                if len(idxes) < 0.5 * self.cfg.update_batch:
                    continue

                weight = data.weight[idxes]

                with self.autocast():
                    new_policy, new_value = self._forward_ac(data.obs[idxes])
                    new_logp = new_policy.log_prob(data.act[idxes])
                    log_ratio = new_logp - data.logp[idxes]
                    ratio = log_ratio.exp()

                    adv_ = data.adv[idxes]
                    if self.cfg.adv_norm:
                        true_adv = adv_.clone()
                        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
                    else:
                        true_adv = adv_

                    t1 = -adv_ * ratio
                    t2 = -adv_ * ratio.clamp(
                        1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                    )
                    policy_loss = (weight * torch.max(t1, t2)).mean()

                    if self.cfg.clip_vloss:
                        clipped_v = data.val[idxes] + (
                            new_value - data.val[idxes]
                        ).clamp(-self.cfg.clip_coef, self.cfg.clip_coef)
                        v_loss1 = (new_value - data.ret[idxes]).square()
                        v_loss2 = (clipped_v - data.ret[idxes]).square()
                        v_loss = 0.5 * (weight * torch.max(v_loss1, v_loss2)).mean()
                    else:
                        v_loss = (
                            0.5
                            * (weight * (new_value - data.ret[idxes]).square()).mean()
                        )

                    new_ent = new_policy.entropy()
                    ent_loss = self.alpha.value * (weight * -new_ent).mean()

                    loss = policy_loss + ent_loss + self.cfg.vf_coef * v_loss

                self.opt.step(loss, self.cfg.clip_grad)
                if self.alpha.adaptive:
                    self.alpha.opt_step(new_ent)

        with torch.no_grad():
            mets = {
                "ratio": ratio.mean(),
                "adv": true_adv.mean(),
                "policy_loss": policy_loss,
                "entropy": new_ent.mean(),
                "v_loss": v_loss,
                "value": data.val.mean(),
            }

        return mets

    def _process_data_equal_size(self, batch: Slices):
        obs = batch.obs[:-1]
        act = batch.act
        reward = self._transform_reward(batch.reward)

        cont = 1.0 - batch.term.float()
        weight = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])
        weight = torch.cumprod(weight, 0)[:-1]

        policy, val = over_seq(self._forward_ac)(batch.obs)
        logp = policy[:-1].log_prob(batch.act)

        gamma = self.cfg.gamma * cont
        adv, ret = gen_adv_est(reward, val, gamma, self.cfg.gae_lambda)
        val = val[:-1]

        obs = obs.flatten(0, 1)
        act = act.flatten(0, 1)
        logp = logp.flatten(0, 1)
        adv = adv.flatten(0, 1)
        ret = ret.flatten(0, 1)
        val = val.flatten(0, 1)
        weight = weight.flatten(0, 1)

        return Data(obs, act, logp, adv, ret, val, weight)

    def _process_data_var_size(self, batch: list[Slices]):
        batch_size = len(batch)
        lengths = torch.tensor([len(seq.obs) for seq in batch])
        end = torch.cumsum(lengths, 0)
        start = end - lengths

        obs = torch.cat([seq.obs[:-1] for seq in batch])
        all_obs = torch.cat([seq.obs for seq in batch])
        act = torch.cat([seq.act for seq in batch])

        policy, value = self._forward_ac(all_obs)

        policy = torch.cat(
            [policy[start[idx] : end[idx] - 1] for idx in range(batch_size)]
        )
        logp = policy.log_prob(act)

        advs, rets, vals, weights = [[] for _ in range(4)]
        for idx, seq in enumerate(batch):
            cont = 1.0 - seq.term.float()
            seq_wt = torch.cat([torch.ones_like(cont[:1]), cont])
            seq_wt = torch.cumprod(seq_wt, 0)

            gamma = self.cfg.gamma * cont
            seq_val = value[start[idx] : end[idx]]
            reward = self._transform_reward(seq.reward)
            adv, ret = gen_adv_est(reward, seq_val, gamma, self.cfg.gae_lambda)
            advs.append(adv)
            rets.append(ret)
            vals.append(seq_val[:-1])
            weights.append(seq_wt[:-1])

        adv = torch.cat(advs)
        ret = torch.cat(rets)
        val = torch.cat(vals)
        weight = torch.cat(weights)

        return Data(obs, act, logp, adv, ret, val, weight)

    def _transform_reward(self, reward: Tensor):
        if self.cfg.rew_fn == "sign":
            return torch.sign(reward)
        elif self.cfg.rew_fn == "tanh":
            return torch.tanh(reward)
        elif self.cfg.rew_fn == "id":
            return reward
