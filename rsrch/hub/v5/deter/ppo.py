from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.exp.api import Experiment
from rsrch.rl import gym
from rsrch.utils import cron

from ..agent.ppo import Config
from ..common import alpha
from ..common.utils import flat, gae_adv_est, over_seq
from .wm import WorldModel


class Actor:
    def __call__(self, state: Tensor) -> D.Distribution:
        """Get a distribution over actions for a batch of states. The distribution
        is over the encoded actions, to make function differentiable when the
        action space is discrete.
        """


class Critic:
    def __call__(self, state: Tensor) -> Tensor:
        """Get state values for a batch of states."""


class Context:
    should_log: cron.Flag
    exp: Experiment


class Trainer:
    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        actor: Actor,
        act_space: gym.TensorSpace,
        make_critic: Callable[[], Critic],
        ctx: Context,
    ):
        self.cfg = cfg
        self.wm = wm
        self.actor = actor
        self.critic = make_critic()
        self.ac_params = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = cfg.opt.make()(self.ac_params)
        self.alpha = alpha.Alpha(cfg.alpha, act_space)
        self.ctx = ctx

    def opt_step(self, trans_hx: Tensor):
        cur_h = self.wm.init_pred(trans_hx)  # [L, N, H]

        cur_h, all_hx = cur_h, [cur_h[-1]]
        act_rvs, acts = [], []
        for _ in range(self.cfg.horizon):
            act_rv = self.actor(cur_h[-1])
            act_rvs.append(act_rv)
            act = act_rv.rsample()
            acts.append(act)
            _, next_h = self.wm.pred(act.unsqueeze(0), cur_h)
            all_hx.append(next_h[0])
            cur_h = next_h

        all_hx = torch.stack(all_hx)  # [L+1, N, H]
        act = torch.stack(acts)  # [L, N, H]
        act_rvs: D.Distribution = torch.stack(act_rvs)  # [L, H]

        with torch.no_grad():
            val = over_seq(self.critic)(all_hx)
            rew = over_seq(self.wm.reward)(all_hx[1:]).mean
            term = over_seq(self.wm.term)(all_hx).mode
            logp = act_rvs.log_prob(act)
            term = 1.0 - (1.0 - term).cumprod(0)
            val = val * term
            adv, ret = gae_adv_est(rew, val, self.cfg.gamma, self.cfg.gae_lambda)
            val, all_hx = val[:-1], all_hx[:-1]

        act = flat(act)
        logp = flat(logp)
        adv = flat(adv)
        ret = flat(ret)
        val = flat(val)
        all_hx = flat(all_hx)

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(len(act))
            for idxes in perm.split(self.cfg.update_batch):
                new_pi = self.actor(all_hx[idxes])
                new_logp = new_pi.log_prob(act[idxes])
                new_val = self.critic(all_hx[idxes])
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
                    clipped_v = val[idxes] + (new_val - val[idxes]).clamp(
                        -self.cfg.clip_coeff, self.cfg.clip_coeff
                    )
                    v_loss1 = (new_val - ret[idxes]).square()
                    v_loss2 = (clipped_v - ret[idxes]).square()
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_val - ret[idxes]).square().mean()
                v_loss = self.cfg.vf_coeff * v_loss

                new_ent = new_pi.entropy()
                ent_loss = self.alpha.value * -new_ent.mean()

                loss = policy_loss + v_loss + ent_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        self.ac_params, self.cfg.clip_grad
                    )
                self.opt.step()

                self.alpha.opt_step(new_ent)

        if self.ctx.should_log:
            exp = self.ctx.exp
            for pfx in ("", "policy", "v", "ent"):
                k = "loss" if pfx == "" else f"{pfx}_loss"
                exp.add_scalar(f"train/{k}", locals()[k])
            exp.add_scalar("train/mean_v", val.mean())
