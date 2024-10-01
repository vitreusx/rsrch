from collections import namedtuple
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode

from ..common import dh, nets
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import find_class


@dataclass
class Config:
    encoder: dict
    actor_dist: dict
    update_epochs: int
    update_batch: int
    adv_norm: bool
    clip_coeff: float
    clip_vloss: bool
    gamma: float
    gae_lambda: float
    opt: dict
    clip_grad: float | None
    ent_coeff: float
    vf_coeff: float
    share_encoder: bool


def gen_adv_est(
    reward: Tensor,
    value: Tensor,
    gamma: float,
    gae_lambda: float,
):
    delta = (reward + gamma * value[1:]) - value[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(reward))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = value[:-1] + adv
    return adv, ret


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


class CriticHead(nn.Sequential):
    def __init__(self, in_features: int):
        super().__init__(
            nn.Linear(in_features, 1),
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


TrainerOutput = namedtuple("TrainerOutput", ("loss", "metrics"))


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
                    obs = batch.obs[:-1]
                    act = batch.act
                    policy, val = over_seq(self._forward_ac)(batch.obs)
                    logp = policy[:-1].log_prob(batch.act)
                    cont = 1.0 - batch.term.float()
                    val, reward = val * cont, batch.reward * cont[:-1]
                    adv, ret = gen_adv_est(
                        reward, val, self.cfg.gamma, self.cfg.gae_lambda
                    )
                    val = val[:-1]

                    tmp_ = (x.flatten(0, 1) for x in (obs, act, logp, adv, ret, val))
                    obs, act, logp, adv, ret, val = tmp_

                else:
                    obs, act, logp, adv, ret, val = [], [], [], [], [], []
                    for seq in batch:
                        obs.append(seq.obs[:-1])
                        act.append(seq.act)
                        policy, value = self._forward_ac(seq.obs)
                        logp_ = policy[:-1].log_prob(seq.act)
                        logp.append(logp_)
                        cont = 1.0 - seq.term.float()
                        value, reward = value * cont, seq.reward * cont[:-1]
                        val.append(value[:-1])
                        adv_, ret_ = gen_adv_est(
                            reward, value, self.cfg.gamma, self.cfg.gae_lambda
                        )
                        adv.append(adv_)
                        ret.append(ret_)

                    tmp_ = (torch.cat(x) for x in (obs, act, logp, adv, ret, val))
                    obs, act, logp, adv, ret, val = tmp_

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(len(val))
            for idxes in perm.split(self.cfg.update_batch):
                if len(idxes) < self.cfg.update_batch:
                    cont

                with self.autocast():
                    new_policy, new_value = self._forward_ac(obs[idxes])
                    new_logp = new_policy.log_prob(act[idxes])
                    log_ratio = new_logp - logp[idxes]
                    ratio = log_ratio.exp()

                    adv_ = adv[idxes]
                    if self.cfg.adv_norm:
                        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)

                    t1 = -adv_ * ratio
                    t2 = -adv_ * ratio.clamp(
                        1 - self.cfg.clip_coeff, 1 + self.cfg.clip_coeff
                    )
                    policy_loss = torch.max(t1, t2).mean()

                    if self.cfg.clip_vloss:
                        clipped_v = val[idxes] + (new_value - val[idxes]).clamp(
                            -self.cfg.clip_coeff, self.cfg.clip_coeff
                        )
                        v_loss1 = (new_value - ret[idxes]).square()
                        v_loss2 = (clipped_v - ret[idxes]).square()
                        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    else:
                        v_loss = 0.5 * (new_value - ret[idxes]).square().mean()

                    new_ent = new_policy.entropy()
                    ent_loss = -new_ent.mean()

                    loss = (
                        policy_loss
                        + self.cfg.ent_coeff * ent_loss
                        + self.cfg.vf_coeff * v_loss
                    )

                self.opt.step(loss, self.cfg.clip_grad)

        with torch.no_grad():
            mets = {
                "ratio": ratio.mean(),
                "adv": adv_.mean(),
                "policy_loss": policy_loss,
                "entropy": new_ent.mean(),
                "ent_loss": ent_loss,
                "v_loss": v_loss,
                "value": val.mean(),
            }

        return mets
