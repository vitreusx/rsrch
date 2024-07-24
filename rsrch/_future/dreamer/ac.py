from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from . import nets, rssm
from .utils import find_class, null_ctx
from .wm import WorldModel


@dataclass
class Config:
    actor: dict
    critic: dict
    opt: dict
    coef: dict
    target_critic: dict | None
    rew_norm: dict
    actor_grad: Literal["dynamics", "reinforce", "auto"]
    horizon: int
    clip_grad: float | None
    gamma: float
    gae_lambda: float
    actor_grad_mix: float | dict
    actor_ent: float | dict


class Actor(nn.Module):
    def __init__(self, wm: WorldModel, cfg: Config):
        super().__init__()
        self.wm = wm
        self.cfg = cfg

        mlp = nets.MLP(wm.state_size, None, **cfg.actor)
        head = nets.ActorHead(mlp.out_features, wm.act_space)
        self.net = nn.Sequential(rssm.AsTensor(), mlp, head)

    def forward(self, state: rssm.State):
        return self.net(state)

    def policy(self, state: rssm.State, sample=True):
        enc_act_dist: D.Distribution = self(state)
        if sample:
            enc_act = enc_act_dist.sample()
        else:
            enc_act = enc_act_dist.mode
        return self.wm.act_enc.inverse(enc_act)


class Critic(nn.Sequential):
    def __init__(self, wm: WorldModel, cfg: Config):
        vf_space = spaces.torch.Box((), dtype=torch.float32)
        super().__init__(
            rssm.AsTensor(),
            nets.BoxDecoder(wm.state_size, vf_space, **cfg.critic),
        )


def gae_lambda(rew, val, gamma, bootstrap, lambda_):
    next_values = torch.cat((val[1:], bootstrap[None]), 0)
    inputs = rew + (1.0 - lambda_) * gamma * next_values

    returns, cur = [], bootstrap
    for t in reversed(range(len(inputs))):
        cur = inputs[t] + lambda_ * gamma[t] * cur
        returns.append(cur)

    returns.reverse()
    return torch.stack(returns)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        compute_dtype: torch.dtype | None,
    ):
        super().__init__()
        self.cfg = cfg
        self.compute_dtype = compute_dtype

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)
        self.actor_grad_mix = self._make_sched(cfg.actor_grad_mix)
        self.actor_ent = self._make_sched(cfg.actor_ent)

    def setup(
        self,
        wm: WorldModel,
        actor: Actor,
        make_critic: Callable[[], nn.Module],
    ):
        self.wm = wm
        self.actor = actor

        self.critic = make_critic()

        if self.cfg.target_critic is not None:
            self.target_critic = make_critic()
            polyak.sync(self.critic, self.target_critic)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.target_critic,
                **self.cfg.target_critic,
            )
        else:
            self.target_critic = self.critic

        if self.cfg.actor_grad == "auto":
            discrete = isinstance(self.wm.act_space, spaces.torch.Discrete)
            self.actor_grad = "reinforce" if discrete else "dynamics"
        else:
            self.actor_grad = self.cfg.actor_grad

        self.opt = self._make_opt()
        self.parameters = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt_iter = 0
        self._device = next(self.actor.parameters()).device
        self.scaler = getattr(torch, self._device.type).amp.GradScaler()

    def _make_opt(self):
        cfg = {**self.cfg.opt}

        cls = find_class(torch.optim, cfg["type"])
        del cfg["type"]

        actor, critic = cfg["actor"], cfg["critic"]
        del cfg["actor"]
        del cfg["critic"]

        common = cfg

        return cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.critic.parameters(), **critic},
            ],
            **common,
        )

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, float):
            return sched.Constant(cfg)
        else:
            cfg = {**cfg}
            cls = getattr(sched, cfg["type"])
            del cfg["type"]
            return cls(**cfg)

    def save(self):
        return {"opt": self.opt.state_dict(), "scaler": self.scaler.state_dict()}

    def load(self, ckpt):
        self.opt.load_state_dict(ckpt["opt"])
        self.scaler.load_state_dict(ckpt["scaler"])

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self._device.type,
                dtype=self.compute_dtype,
            )

    def compute(self, batch: dict):
        losses, mets = {}, {}

        if self.update_target is not None:
            self.update_target.step()

        with self.autocast():
            state = batch["initial"]
            states, acts = [state], []
            for _ in range(self.cfg.horizon):
                policy: D.Distribution = self.actor(state.detach())
                act: Tensor = policy.rsample()
                acts.append(act)
                state = self.wm.rssm.img_step(state, act)
                states.append(state)
            states, acts = torch.stack(states), torch.stack(acts)

        # For reinforce, we detach `target` variable, so requiring gradients on
        # any computation leading up to it will cause autograd to leak memory
        no_grad = torch.no_grad if self.actor_grad == "reinforce" else null_ctx
        with no_grad():
            with self.autocast():
                rew_dist = over_seq(self.wm.decoders["reward"])(states)
                reward = self.rew_norm(rew_dist.mode)

                term_dist = over_seq(self.wm.decoders["term"])(states)
                term = term_dist.mean
                term[0] = batch["term"].float()
                gamma = self.cfg.gamma * (1.0 - term)

                vt = over_seq(self.target_critic)(states).mode
                target = gae_lambda(
                    reward[:-1], vt[:-1], gamma[:-1], vt[-1], self.cfg.gae_lambda
                )

        with torch.no_grad():
            with self.autocast():
                weight = torch.cat([torch.ones_like(gamma[:1]), gamma[:-1]])
                weight = weight.cumprod(0)

        with self.autocast():
            policies = self.actor(states[:-2].detach())
            if self.actor_grad == "dynamics":
                objective = target[1:]
            elif self.actor_grad == "reinforce":
                baseline = over_seq(self.target_critic)(states[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(acts[:-1].detach())
            elif self.actor_grad == "both":
                baseline = over_seq(self.target_critic)(states[:-2]).mode
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(acts[:-1].detach())
                mix = self.actor_grad_mix(self.opt_iter)
                mets["actor_grad_mix"] = mix
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.actor_ent(self.opt_iter)
            mets["policy_ent_scale"] = ent_scale
            policy_ent = policies.entropy()
            mets["policy_ent"] = policy_ent.mean()
            objective = objective + ent_scale * policy_ent
            actor_loss = -(weight[:-2] * objective).mean()
            mets["actor_loss"] = actor_loss

            value_dist = over_seq(self.critic)(states[:-1].detach())
            mets["value"] = (value_dist.mean).mean()
            critic_loss = (weight[:-1] * -value_dist.log_prob(target)).mean()
            mets["critic_loss"] = critic_loss

            for k, v in losses:
                mets[f"{k}_loss"] = v.detach()

            coef = self.cfg.coef
            loss = sum(coef.get(k, 1.0) * v for k, v in losses.items())

        return loss, mets

    def opt_step(self, loss: Tensor):
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        if self.cfg.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters, max_norm=self.cfg.clip_grad)
        self.scaler.step(self.opt)
        self.scaler.update()
