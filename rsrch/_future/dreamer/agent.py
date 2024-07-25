from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch._future import rl

from .ac import Actor
from .utils import autocast
from .wm import WorldModel


@dataclass
class Config:
    expl_noise: float
    eval_noise: float


class Agent(rl.VecAgent):
    def __init__(
        self,
        actor: Actor,
        cfg: Config,
        act_space: spaces.torch.Space,
        mode: Literal["prefill", "expl", "eval"] = "prefill",
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.actor = actor
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

            with torch.inference_mode():
                with autocast(self._device, self.compute_dtype):
                    return func(self, *args, **kwargs)

        return wrapped

    @compute_ctx
    def reset(self, idxes, obs: Tensor):
        obs = obs.to(self._device)
        state = self.actor.wm.reset(obs)
        if self._state is None:
            self._state = state
        else:
            self._state[idxes] = state.type_as(self._state)

    @compute_ctx
    def policy(self, idxes):
        if self.mode == "prefill":
            act = self.act_space.sample([len(idxes)])
        else:
            if self.mode == "expl":
                sample = True
                noise = self.cfg.expl_noise
            elif self.mode == "eval":
                sample = False
                noise = self.cfg.eval_noise
            act = self.actor.policy(self._state[idxes], sample)
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
        state = self.actor.wm.step(self._state[idxes], act, next_obs)
        self._state[idxes] = state.type_as(self._state)
