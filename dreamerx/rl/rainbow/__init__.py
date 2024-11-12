from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import safe_mode
from rsrch.rl import gym
from rsrch.rl.utils import polyak

from ...common import nets
from ...common.trainer import ScaledOptimizer, TrainerBase
from ...common.types import Slices
from ...common.utils import autocast, to_camel_case
from . import config, noisy
from .dist_q import ValueDist


class QHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_space: spaces.torch.Discrete,
        hidden_dim: int,
        dist: config.Dist,
        dueling: bool,
    ):
        super().__init__()
        self.in_features = in_features
        self.act_space = act_space
        self.hidden_dim = hidden_dim
        self.dist = dist
        self.dueling = dueling

        mult = dist.num_atoms if dist.enabled else 1
        if dueling:
            self.v_head = self._make_head(mult)
            self.adv_head = self._make_head(mult * act_space.n)
        else:
            self.q_head = self._make_head(mult * act_space.n)

    def _make_head(self, out_features: int):
        return nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_features),
        )

    def forward(self, input: Tensor):
        if self.dist.enabled:
            if self.dueling:
                v: Tensor = self.v_head(input)
                v = v.reshape(len(v), 1, self.dist.num_atoms)
                adv: Tensor = self.adv_head(input)
                adv = adv.reshape(len(adv), self.act_space.n, self.dist.num_atoms)
                logits = v + adv - adv.mean(-2, keepdim=True)
            else:
                logits: Tensor = self.q_head(input)
                logits = logits.reshape(-1, self.act_space.n, self.dist.num_atoms)

            qv = ValueDist(
                ind_rv=D.Categorical(logits=logits),
                v_min=self.dist.v_min,
                v_max=self.dist.v_max,
            )

        else:
            if self.dueling:
                adv: Tensor = self.adv_head(input)
                v: Tensor = self.v_head(input)
                qv = v.flatten(1) + adv - adv.mean(-1, keepdim=True)
            else:
                qv = self.q_head(input)

        return qv


class Q(nn.Module):
    def __init__(
        self,
        cfg: config.Config,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Discrete,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_space = obs_space
        self.act_space = act_space

        self.encoder = nets.make_encoder(obs_space, **cfg.encoder)
        with safe_mode(self.encoder):
            input = obs_space.sample()
            z_features = self.encoder(input).shape[1]

        self.head = QHead(
            z_features,
            act_space,
            cfg.hidden_dim,
            cfg.dist,
            cfg.dueling,
        )

        if cfg.noisy.enabled:
            noisy.replace_all(self, cfg.noisy.sigma0, cfg.noisy.factorized)

    def forward(self, obs: Tensor):
        return self.head(self.encoder(obs))


class Agent(gym.vector.agents.Markov):
    def __init__(
        self,
        qf: Q,
        mode: Literal["train", "eval"],
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(qf.obs_space, qf.act_space)
        self.qf = qf
        self.mode = mode
        self.compute_dtype = compute_dtype
        self._noisy_mode = "zero" if mode == "eval" else "reset"

    @cached_property
    def device(self):
        return next(self.qf.parameters()).device

    @contextmanager
    def compute_ctx(self):
        if self.q.training:
            self.q.eval()

        with torch.no_grad():
            with autocast(self.device, self.compute_dtype):
                yield

    def get_policy(self, last_obs: Tensor):
        with self.compute_ctx():
            last_obs = last_obs.to(self.device)
            with noisy.on_forward(self._noisy_mode):
                qv = self.q(last_obs)
            if isinstance(qv, ValueDist):
                qv = qv.mean
            act = qv.argmax(-1)
        return act


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: config.Config,
        qf: Q,
        is_coef_exp: Callable[[], float] | None = None,
        prio_exp: Callable[[], float] | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg

        self.qf = qf
        self.qf_t = Q(cfg, qf.obs_space, qf.act_space)
        polyak.sync(self.qf, self.qf_t)
        self.qf_polyak = polyak.Polyak(self.qf, self.qf_t, **cfg.polyak)

        self.opt = self._make_opt(self.qf.parameters(), cfg.opt)
        self.is_coef_exp = is_coef_exp
        self.prio_exp = prio_exp

    def _make_opt(self, parameters: list[nn.Parameter], cfg: dict):
        cls = getattr(torch.optim, to_camel_case(cfg["type"]))
        del cfg["type"]
        opt = cls(parameters, **cfg)
        opt = ScaledOptimizer(opt)
        return opt

    @dataclass
    class Output:
        metrics: dict
        prio_values: Tensor

    def opt_step(self, batch: Slices, is_coefs: Tensor | None = None):
        rew = self._reward_fn(batch.reward)
        slice_len = batch.obs.shape[0]

        with torch.no_grad():
            with self.autocast():
                next_q_eval = self.qf_t(batch.obs[-1])

                if isinstance(next_q_eval, ValueDist):
                    if self.cfg.double_dqn:
                        next_q_act: ValueDist = self.qf(batch.obs[-1])
                    else:
                        next_q_act = next_q_eval
                    act = next_q_act.mean.argmax(-1)
                    target = next_q_eval.gather(-1, act[..., None])
                    target = target.squeeze(-1)
                elif isinstance(next_q_eval, Tensor):
                    if self.cfg.double_dqn:
                        next_q_act: Tensor = self.qf(batch.obs[-1])
                        act = next_q_act.argmax(-1)
                        target = next_q_eval.gather(-1, act[..., None])
                        target = target.squeeze(-1)
                    else:
                        target = next_q_eval.max(-1).values

                cont = 1.0 - batch.term.float()
                target = cont[-1] * target
                for t in reversed(range(slice_len - 2)):
                    target = rew[t] + self.cfg.gamma * target

        with self.autocast():
            qv: Tensor | ValueDist = self.qf(batch.obs[0])
            pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

            if isinstance(target, ValueDist):
                q_losses = ValueDist.w1_div(target, pred)
            else:
                q_losses = (pred - target).square()

            if is_coefs is not None:
                weights = is_coefs ** self.is_coef_exp()
                q_losses = weights * q_losses

            loss = q_losses.mean()

        self.opt.step(loss, self.cfg.clip_grad)

        with torch.no_grad():
            pred = pred.mean if isinstance(pred, ValueDist) else pred

            if isinstance(target, ValueDist):
                prio = q_losses.detach()
            else:
                prio = (pred - target).square()
            prio = prio ** self.prio_exp()

            mets = {
                "loss": loss,
                "mean_q_pred": pred.mean(),
                "target": target.mean(),
            }

        return Trainer.Output(mets, prio)

    def _reward_fn(self, reward: Tensor):
        rew_fn = self.cfg.rew_fn
        if rew_fn == "clip":
            return reward.clamp(*self.cfg.rew_clip)
        elif rew_fn == "tanh":
            return reward.tanh()
        else:
            return reward
