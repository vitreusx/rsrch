from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, Literal, NamedTuple, TypedDict

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer

import rsrch.torch.distributions as D
from rsrch import spaces
from rsrch.rl.utils import polyak
from rsrch.rl.utils.gae import gen_adv_est
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.torch.nn.utils import over_seq


class Policy(D.Distribution, Tensor):
    """Action distribution interface.

    NOTE: Regular `torch.distributions` objects do not have reshaping/indexing ops necessary. You can use `rsrch.distributions` instead, or implement appropriate distribution classes yourself.
    """


class Actor(nn.Module):
    """Actor interface."""

    def __call__(self, state: Tensor) -> Policy:
        """Get policy :math:`\pi(a_t \mid s_t)` for a given state."""

    def compute_features(self, state: Tensor) -> Tensor:
        """(Optional) Extract intermediate features, to be shared with critic(s)."""

    def forward_features(self, features: Tensor) -> Policy:
        """(Optional) Get policy :math:`\pi(a_t \mid \phi(s_t))` from state features :math:`\phi(s_t)`"""


class Critic(nn.Module):
    """Critic interface."""

    def __call__(self, state_or_feat: Tensor) -> Tensor:
        """Get value estimates :math:`V(s_t)` from states."""


class Alpha:
    """Interface for :math:`\alpha` entropy coefficient."""

    def opt_step(self, entropy: float):
        ...

    def __float__(self):
        ...


class Slices(TypedDict):
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor


class Data(NamedTuple):
    obs: Tensor
    act: Tensor
    logp: Tensor
    adv: Tensor
    ret: Tensor
    val: Tensor
    weight: Tensor


class Trainer:
    def __init__(
        self,
        actor: Actor,
        make_critic: Callable[[], Critic],
        make_actor_opt: Callable[[list[Parameter]], Optimizer],
        make_critic_opt: Callable[[list[Parameter]], Optimizer],
        update_epochs: int,
        mb_size: int | None,
        adv_norm: bool,
        clip_coef: float,
        clip_vloss: bool,
        gamma: float,
        gae_lambda: float,
        clip_grad: float | None,
        vf_coef: float,
        rew_norm: Callable[[Tensor], Tensor] | None,
        alpha: float | Alpha,
        target_critic: polyak.Config | None,
        share_encoder: bool,
        compute_dtype: torch.dtype,
    ):
        self.update_epochs = update_epochs
        self.mb_size = mb_size
        self.adv_norm = adv_norm
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_grad = clip_grad
        self.vf_coef = vf_coef
        self.rew_norm = rew_norm
        self.share_encoder = share_encoder

        self.actor = actor
        self.critic = make_critic()

        self.actor_opt = make_actor_opt(self.actor.parameters())
        self.actor_opt = ScaledOptimizer(self.actor_opt)
        self.critic_opt = make_critic_opt(self.critic.parameters())
        self.critic_opt = ScaledOptimizer(self.critic_opt)

        self.alpha = alpha

        if target_critic is not None:
            self.critic_t = make_critic()
            self.critic_t.requires_grad_(False)

            polyak.sync(
                source=self.critic,
                target=self.critic_t,
            )
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.critic_t,
                tau=target_critic.tau,
                every=target_critic.every,
            )

        device = next(self.actor.parameters()).device
        self.autocast = lambda: torch.autocast(device.type, compute_dtype)

    def opt_step(
        self,
        batch: Slices | list[Slices],
        compute_metrics: bool = False,
    ):
        with torch.no_grad():
            with self.autocast():
                if isinstance(batch, Slices):
                    data = self._process_data_equal_size(batch)
                else:
                    data = self._process_data_var_size(batch)

        for _ in range(self.update_epochs):
            if self.mb_size is None:
                splits = [slice(0, len(data.val))]
            else:
                batch_size = len(data.val)
                num_slices = batch_size // self.mb_size
                pivots = [*(i * self.mb_size for i in range(num_slices)), batch_size]
                slices = [slice(p, q) for p, q in zip(pivots, pivots[1:])]
                perm = torch.randperm(batch_size)
                splits = [perm[idx] for idx in slices]

            for idxes in splits:
                weight = data.weight[idxes]

                with self.autocast():
                    if self.share_encoder:
                        features = over_seq(self.actor.compute_features)(batch["obs"])
                        new_policy = over_seq(self.actor.forward_features)(features)
                        new_val = over_seq(self.critic)(features)
                    else:
                        new_policy = over_seq(self.actor)(batch["obs"])
                        new_val = over_seq(self.critic)(batch["obs"])
                    new_logp = new_policy.log_prob(data.act[idxes])
                    log_ratio = new_logp - data.logp[idxes]
                    ratio = log_ratio.exp()

                    adv_ = data.adv[idxes]
                    if self.adv_norm:
                        true_adv = adv_.clone()
                        adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)
                    else:
                        true_adv = adv_

                    t1 = -adv_ * ratio
                    t2 = -adv_ * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                    policy_loss = (weight * torch.max(t1, t2)).mean()

                    new_ent = new_policy.entropy()
                    ent_loss = float(self.alpha) * (weight * -new_ent).mean()

                    actor_loss = policy_loss + ent_loss

                    if self.clip_vloss:
                        clipped_v = data.val[idxes] + (new_val - data.val[idxes]).clamp(
                            -self.clip_coef, self.clip_coef
                        )
                        v_losses1 = (new_val - data.ret[idxes]).square()
                        v_losses2 = (clipped_v - data.ret[idxes]).square()
                        v_losses = torch.max(v_losses1, v_losses2)
                    else:
                        v_losses = (new_val - data.ret[idxes]).square()
                    v_loss = self.vf_coef * (weight * v_losses).mean()

                self.actor_opt.step(actor_loss, clip_grad=self.clip_grad)
                self.critic_opt.step(v_loss, clip_grad=self.clip_grad)

                if self.critic_t is not None:
                    self.update_target.step()

                if hasattr(self.alpha, "opt_step"):
                    self.alpha.opt_step(new_ent)

        result = {}

        if compute_metrics:
            with torch.no_grad():
                result["metrics"] = {
                    "ratio": ratio.mean(),
                    "adv": true_adv.mean(),
                    "policy_loss": policy_loss.detach(),
                    "actor_loss": actor_loss.detach(),
                    "entropy": new_ent.mean(),
                    "v_loss": v_loss.detach(),
                    "value": data.val.mean(),
                }

        return result

    def _process_data_equal_size(self, batch: Slices):
        obs = batch["obs"][:-1]
        act = batch["act"]
        reward = batch["reward"]
        if self.rew_norm is not None:
            reward = self.rew_norm(reward)

        cont = 1.0 - batch["term"].float()
        weight = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])
        weight = torch.cumprod(weight, 0)[:-1]

        if self.share_encoder:
            features = over_seq(self.actor.compute_features)(batch["obs"])
            policy = over_seq(self.actor.forward_features)(features)
            val = over_seq(self.critic)(features)
            if self.critic_t is None:
                val_t = val
            else:
                val_t = over_seq(self.critic_t)(features)
        else:
            policy = over_seq(self.actor)(batch["obs"])
            val = over_seq(self.critic)(batch["obs"])
            if self.critic_t is None:
                val_t = val
            else:
                val_t = over_seq(self.critic_t)(batch["obs"])

        logp: Tensor = policy[:-1].log_prob(batch["act"])

        gamma = self.gamma * cont
        adv, ret = gen_adv_est(reward, val_t, gamma, self.gae_lambda)
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
        lengths = torch.tensor([len(seq["obs"]) for seq in batch])
        end = torch.cumsum(lengths, 0)
        start = end - lengths

        obs = torch.cat([seq["obs"][:-1] for seq in batch])
        all_obs = torch.cat([seq["obs"] for seq in batch])
        act = torch.cat([seq["act"] for seq in batch])

        if self.share_encoder:
            features = self.actor.compute_features(all_obs)
            policy = self.actor.forward_features(features)
            val = self.critic(features)
            if self.critic_t is None:
                val_t = val
            else:
                val_t = self.critic_t(features)
        else:
            policy = self.actor(all_obs)
            val = self.critic(all_obs)
            if self.critic_t is None:
                val_t = val
            else:
                val_t = self.critic_t(all_obs)

        policy: Policy = torch.cat(
            [policy[start[idx] : end[idx] - 1] for idx in range(batch_size)]
        )
        logp = policy.log_prob(act)

        advs, rets, vals, weights = [[] for _ in range(4)]
        for idx, seq in enumerate(batch):
            cont = 1.0 - seq["term"].float()
            seq_wt = torch.cat([torch.ones_like(cont[:1]), cont])
            seq_wt = torch.cumprod(seq_wt, 0)

            gamma = self.gamma * cont
            seq_val_t = val_t[start[idx] : end[idx]]
            reward = seq["reward"]
            if self.rew_norm is not None:
                reward = self.rew_norm(reward)
            adv, ret = gen_adv_est(reward, seq_val_t, gamma, self.gae_lambda)
            advs.append(adv)
            rets.append(ret)
            seq_val = val[start[idx] : end[idx]]
            vals.append(seq_val[:-1])
            weights.append(seq_wt[:-1])

        adv = torch.cat(advs)
        ret = torch.cat(rets)
        val = torch.cat(vals)
        weight = torch.cat(weights)

        return Data(obs, act, logp, adv, ret, val, weight)
