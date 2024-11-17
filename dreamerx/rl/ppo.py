from collections import namedtuple
from dataclasses import dataclass
from functools import cache, partial
from typing import Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl.utils import polyak

from ..common import nets
from ..common.trainer import ScaledOptimizer, TrainerBase
from ..common.types import Slices
from ..common.utils import find_class
from . import _alpha as alpha
from ._utils import gen_adv_est


@dataclass
class Config:
    @dataclass
    class Actor:
        encoder: dict
        dist: dict
        opt: dict

    @dataclass
    class Critic:
        encoder: dict | None
        dist: dict
        opt: dict

    actor: Actor
    critic: Critic
    update_epochs: int
    num_minibatches: int
    adv_norm: bool
    clip_coef: float
    clip_vloss: bool
    gamma: float
    gae_lambda: float
    clip_grad: float | None
    alpha: alpha.Config
    vf_coef: float
    rew_fn: Literal["id", "sign", "tanh"]
    target_critic: dict | None


class Actor(nn.Module):
    def __init__(
        self,
        cfg: Config.Actor,
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
        self.head = dh.make(layer_ctor, act_space, **cfg.dist)

    def forward(self, state: Tensor) -> D.Distribution:
        return self.head(self.encoder(state))

    def forward_features(self, state: Tensor):
        features = self.encoder(state)
        return self.head(features), features


class Critic(nn.Module):
    def __init__(
        self,
        cfg: Config.Critic,
        actor: Actor,
    ):
        super().__init__()

        if cfg.encoder is not None:
            self.encoder = nets.make_encoder(actor.obs_space, **cfg.encoder)
            with safe_mode(self.encoder):
                input = actor.obs_space.sample((1,))
                z_features = self.encoder(input).shape[1]
        else:
            self.encoder = None
            z_features = actor.z_features

        layer_ctor = partial(nn.Linear, z_features)
        value_space = spaces.torch.Tensor(shape=())
        self.head = dh.make(layer_ctor, value_space, **cfg.dist)

    def forward(self, state: Tensor):
        if self.encoder is not None:
            state = self.encoder(state)
        return self.head(state).mean


class Data(NamedTuple):
    obs: Tensor
    act: Tensor
    logp: Tensor
    adv: Tensor
    ret: Tensor
    val: Tensor
    weight: Tensor


@cache
def _get_slices(batch_size: int, num_mb: int):
    """Divide a batch into a number of minibatches. The minibatch sizes are selected in such a way, that they are divisible by 32, except for the last one. The last batch may be larger than the previous ones."""
    WARP = 32
    batch_size_w = batch_size // WARP
    mb_size = WARP * (batch_size_w // num_mb)
    mb_size_rem = batch_size - num_mb * mb_size
    split_sizes = [mb_size] * num_mb
    split_sizes[-1] += mb_size_rem
    end = np.cumsum(split_sizes)
    start = end - np.array(split_sizes)
    return [slice(start_i, end_i) for start_i, end_i in zip(start, end)]


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

        self.critic = Critic(cfg.critic, actor).to(device)

        if self.cfg.target_critic is not None:
            self.critic_t = Critic(cfg.critic, actor).to(device)
            polyak.sync(self.critic, self.critic_t)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.critic_t,
                **self.cfg.target_critic,
            )
        else:
            self.critic_t = None

        self.opt = self._make_opt(
            [
                {"params": self.actor.parameters(), "cfg": cfg.actor.opt},
                {"params": self.critic.parameters(), "cfg": cfg.critic.opt},
            ]
        )

        self.alpha = alpha.Alpha(cfg.alpha, actor.act_space, device)

    def _make_opt(self, groups):
        types = [group["cfg"]["type"] for group in groups]
        assert all(type == types[0] for type in types)
        cls = find_class(torch.optim, types[0])

        opt_groups = []
        for group in groups:
            cfg = {**group["cfg"]}
            del cfg["type"]
            opt_groups.append({"params": group["params"], **cfg})

        opt = cls(opt_groups)
        return ScaledOptimizer(opt)

    def _forward_ac(self, obs: Tensor):
        if self.critic.encoder is None:
            policy, features = self.actor.forward_features(obs)
            val = self.critic(features)
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
            if self.cfg.num_minibatches == 1:
                splits = [slice(len(data.val))]
            else:
                batch_size = len(data.val)
                slices = _get_slices(batch_size, self.cfg.num_minibatches)
                perm = torch.randperm(batch_size)
                splits = [perm[idx] for idx in slices]

            for idxes in splits:
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
                        v_losses1 = (new_value - data.ret[idxes]).square()
                        v_losses2 = (clipped_v - data.ret[idxes]).square()
                        v_losses = torch.max(v_losses1, v_losses2)
                    else:
                        v_losses = (new_value - data.ret[idxes]).square()
                    v_loss = self.cfg.vf_coef * (weight * v_losses).mean()

                    new_ent = new_policy.entropy()
                    ent_loss = self.alpha.value * (weight * -new_ent).mean()

                    loss = policy_loss + ent_loss + v_loss

                self.opt.step(loss, self.cfg.clip_grad)
                if self.critic_t is not None:
                    self.update_target.step()
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

        if self.critic_t is None:
            val_t = val
        else:
            if self.critic_t.encoder is None:
                features = over_seq(self.actor)(batch.obs)
                val_t = over_seq(self.critic_t)(features)
            else:
                val_t = over_seq(self.critic_t)(batch.obs)

        gamma = self.cfg.gamma * cont
        adv, ret = gen_adv_est(reward, val_t, gamma, self.cfg.gae_lambda)
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

        if self.critic_t is None:
            value_t = value
        else:
            if self.critic_t.encoder is None:
                features = self.actor(all_obs)
                value_t = self.critic_t(features)
            else:
                value_t = self.critic_t(all_obs)

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
            seq_val_t = value_t[start[idx] : end[idx]]
            reward = self._transform_reward(seq.reward)
            adv, ret = gen_adv_est(reward, seq_val_t, gamma, self.cfg.gae_lambda)
            advs.append(adv)
            rets.append(ret)
            seq_val = value[start[idx] : end[idx]]
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
