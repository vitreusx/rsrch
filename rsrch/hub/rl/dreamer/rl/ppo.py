from collections import namedtuple
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch.nn.utils import over_seq

from ..common import dh
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import find_class


@dataclass
class Config:
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


class Encoder(nn.Sequential):
    def __init__(self, obs_space: spaces.torch.Tensor):
        if isinstance(obs_space, spaces.torch.Image):
            num_channels = obs_space.shape[0]
            super().__init__(
                nn.Conv2d(num_channels, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
            )
            self.out_features = 512

        elif isinstance(obs_space, spaces.torch.Box):
            obs_dim = int(np.prod(obs_space.shape))
            super().__init__(
                nn.Flatten(),
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.out_features = 64

        else:
            raise ValueError(type(obs_space))


class CriticHead(nn.Sequential):
    def __init__(self, in_features: int):
        super().__init__(
            nn.Linear(in_features, 1),
            nn.Flatten(0),
        )


class ActorCritic(nn.Module):
    def __init__(
        self,
        cfg: Config,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        self.share_encoder = cfg.share_encoder

        if self.share_encoder:
            self.enc = Encoder(obs_space)
            layer_ctor = partial(nn.Linear, self.enc.out_features)
            self.actor_head = dh.make(layer_ctor, act_space, **cfg.actor_dist)
            self.critic_head = CriticHead(self.enc.out_features)
        else:
            actor_enc = Encoder(obs_space)
            layer_ctor = partial(nn.Linear, self.enc.out_features)
            actor_head = dh.make(layer_ctor, act_space)
            self.actor = nn.Sequential(actor_enc, actor_head)

            critic_enc = Encoder(obs_space)
            critic_head = CriticHead(critic_enc.out_features)
            self.critic = nn.Sequential(critic_enc, critic_head)

    def forward(self, state: Tensor, with_values: bool = True):
        if self.share_encoder:
            z = self.enc(state)
            policy = self.actor_head(z)
        else:
            policy = self.actor(state)

        if with_values:
            if self.share_encoder:
                value = self.critic_head(z)
            else:
                value = self.critic(state)
            return policy, value
        else:
            return policy


TrainerOutput = namedtuple("TrainerOutput", ("loss", "metrics"))


class Trainer(TrainerBase):
    ON_POLICY = True

    def __init__(
        self,
        cfg: Config,
        ac: ActorCritic,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(clip_grad=None, compute_dtype=compute_dtype)
        self.cfg = cfg
        self.ac = ac
        self.opt = self._make_opt()

    def _make_opt(self):
        cfg = {**self.cfg.opt}
        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]
        opt = cls(self.ac.parameters(), **cfg)
        return ScaledOptimizer(opt)

    def opt_step(self, batch: Slices | list[Slices]):
        with self.autocast():
            if isinstance(batch, list):
                obs, act, logp, adv, ret, val = [], [], [], [], [], []
                for seq in batch:
                    obs.append(seq.obs[:-1])
                    act.append(seq.act)
                    policy, value = self.ac(seq.obs)
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
            else:
                obs = batch.obs[:-1]
                act = batch.act
                policy, val = over_seq(self.ac)(batch.obs)
                logp = policy[:-1].log_prob(batch.act)
                cont = 1.0 - batch.term.float()
                val, reward = val * cont, batch.reward * cont[:-1]
                adv, ret = gen_adv_est(reward, val, self.cfg.gamma, self.cfg.gae_lambda)
                val = val[:-1]

                tmp_ = (x.flatten(0, 1) for x in (obs, act, logp, adv, ret, val))
                obs, act, logp, adv, ret, val = tmp_

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(len(val))
            for idxes in perm.split(self.cfg.update_batch):
                with self.autocast():
                    new_policy, new_value = self.ac(obs[idxes])
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
                        clipped_v = value[idxes] + (new_value - value[idxes]).clamp(
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

        return TrainerOutput(loss, mets)
