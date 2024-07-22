from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch._future import rl

from .wm import WorldModel


class Actor:
    wm: WorldModel

    def policy(self, state) -> D.Distribution:
        ...


@dataclass
class Config:
    expl_noise: float
    eval_noise: float


class Agent(rl.VecAgent):
    def __init__(
        self,
        actor: Actor,
        cfg: Config,
        mode: Literal["prefill", "expl", "eval"],
        act_space: spaces.torch.Space,
    ):
        super().__init__()
        self.actor = actor
        self.cfg = cfg
        self.mode = mode
        self.act_space = act_space
        self._state = None

    def reset(self, idxes, obs: Tensor):
        state = self.actor.wm.reset(obs)
        if self._state is None:
            self._state = state
        else:
            self._state[idxes] = state.type_as(self._state)

    def policy(self, idxes):
        if self.mode == "prefill":
            return self.act_space.sample([len(idxes)])
        else:
            act_dist: D.Distribution = self.actor.policy(self._state[idxes])
            if self.mode == "expl":
                act = act_dist.sample()
                noise = self.cfg.expl_noise
            elif self.mode == "eval":
                act = act_dist.mode
                noise = self.cfg.eval_noise
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

    def step(self, idxes, act: Tensor, next_obs: Tensor):
        state = self.actor.wm.step(self._state[idxes], act, next_obs)
        self._state[idxes] = state.type_as(self._state)
