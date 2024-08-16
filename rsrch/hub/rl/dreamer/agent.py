from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch._future import rl

from .common import nets
from .common.utils import autocast


@dataclass
class Config:
    train_noise: float
    eval_noise: float


class Actor(nn.Module):
    def __call__(self, state) -> D.Distribution:
        ...


class WorldModel(nn.Module):
    act_enc: nets.ActionEncoder

    def reset(self, obs: Tensor) -> Tensor:
        ...

    def step(self, state: Tensor, act: Tensor, next_obs: Tensor) -> Tensor:
        ...


class Agent(rl.VecAgent):
    def __init__(
        self,
        actor: Actor,
        wm: WorldModel,
        cfg: Config,
        act_space: spaces.torch.Tensor,
        mode: Literal["rand", "train", "eval"] = "rand",
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.actor = actor
        self.wm = wm
        self.cfg = cfg
        self.act_space = act_space
        self.mode = mode
        self.compute_dtype = compute_dtype
        self._device = next(self.actor.parameters()).device
        self._state = None

    @staticmethod
    def compute_ctx(func):
        @wraps(func)
        def wrapped(self: "Agent", *args, **kwargs):
            if self.actor.training:
                self.actor.eval()
            if self.wm.training:
                self.wm.eval()

            with torch.no_grad():
                with autocast(self._device, self.compute_dtype):
                    return func(self, *args, **kwargs)

        return wrapped

    @compute_ctx
    def reset(self, idxes, obs: Tensor):
        obs = obs.to(self._device)
        state = self.wm.reset(obs)
        if self._state is None:
            self._state = state
        else:
            self._state[idxes] = state.type_as(self._state)

    @compute_ctx
    def policy(self, idxes):
        if self.mode == "rand":
            act = self.act_space.sample([len(idxes)])
        else:
            if self.mode == "train":
                sample = True
                noise = self.cfg.train_noise
            elif self.mode == "eval":
                sample = False
                noise = self.cfg.eval_noise
            policy = self.actor(self._state[idxes])
            enc_act = policy.sample() if sample else policy.mode
            act = self.wm.act_enc.inverse(enc_act)
            act = act.cpu()
            act = self._apply_noise(act, noise)
        return act

    def _apply_noise(self, act: Tensor, noise: float):
        if noise > 0.0:
            n = len(act)
            if isinstance(self.act_space, spaces.torch.Discrete):
                rand_act = self.act_space.sample((n,)).type_as(act)
                use_rand = (torch.rand(n) < noise).to(act.device)
                act = torch.where(use_rand, rand_act, act)
            elif isinstance(self.act_space, spaces.torch.Box):
                eps = torch.randn(self.act_space.shape).type_as(act)
                low = self.act_space.low.type_as(act)
                high = self.act_space.high.type_as(act)
                act = (act + noise * eps).clamp(low, high)
        return act

    @compute_ctx
    def step(self, idxes, act: Tensor, next_obs: Tensor):
        act, next_obs = act.to(self._device), next_obs.to(self._device)
        state = self.wm.step(self._state[idxes], act, next_obs)
        self._state[idxes] = state.type_as(self._state)
