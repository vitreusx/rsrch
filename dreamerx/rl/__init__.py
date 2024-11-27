from contextlib import contextmanager
from functools import cached_property
from typing import Any

import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.rl import gym

from ..common.utils import autocast


class Actor:
    obs_space: Any
    act_space: Any

    def __call__(self, state: Tensor) -> D.Distribution: ...


class Agent(gym.vector.agents.Markov):
    def __init__(
        self,
        actor: Actor,
        sample: bool = True,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(actor.obs_space, actor.act_space)
        self.actor = actor
        self.sample = sample
        self.compute_dtype = compute_dtype

    @cached_property
    def device(self):
        return next(self.actor.parameters()).device

    @contextmanager
    def compute_ctx(self):
        if self.actor.training:
            self.actor.eval()

        with torch.no_grad():
            with autocast(self.device, self.compute_dtype):
                yield

    def get_policy(self, last_obs: Tensor):
        with self.compute_ctx():
            last_obs = last_obs.to(self.device)
            policy: D.Distribution = self.actor(last_obs)
            act = policy.sample() if self.sample else policy.mode
        return act
