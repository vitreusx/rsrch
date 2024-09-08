from contextlib import contextmanager
from functools import cached_property
from typing import Any, Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym

from ..common.nets import ActionEncoder
from ..common.utils import autocast


class WorldModel(nn.Module):
    act_enc: ActionEncoder
    reward_dec: Callable[[Tensor], D.Distribution]
    term_dec: Callable[[Tensor], D.Distribution]

    def reset(
        self,
        obs: Tensor,
    ) -> D.Distribution:
        ...

    def obs_step(
        self,
        state: Tensor,
        act: Tensor,
        next_obs: Tensor,
    ) -> D.Distribution:
        ...

    def img_step(
        self,
        state: Tensor,
        act: Tensor,
    ) -> D.Distribution:
        ...

    def observe(
        self,
        input: tuple[Tensor, Tensor],
        h_0: list[Tensor | None],
    ) -> tuple[Any, Tensor]:
        ...


class Agent(gym.VecAgentWrapper):
    def __init__(
        self,
        agent: gym.VecAgent,
        wm: WorldModel,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(agent)
        self.wm = wm
        self.compute_dtype = compute_dtype
        self._state = None

    @cached_property
    def device(self):
        return next(self.wm.parameters()).device

    @contextmanager
    def compute_ctx(self):
        if self.wm.training:
            self.wm.eval()

        with torch.no_grad():
            with autocast(self.device, self.compute_dtype):
                yield

    def reset(self, idxes, obs: Tensor):
        obs = obs.to(self.device)
        with self.compute_ctx():
            dist = self.wm.reset(obs)
            state = dist.sample()
        super().reset(idxes, state)
        if self._state is None:
            self._state = state.clone()
        else:
            self._state[idxes] = state.type_as(self._state)

    def policy(self, idxes):
        enc_act = super().policy(idxes)
        act = self.wm.act_enc.inverse(enc_act)
        return act

    def step(self, idxes, act: Tensor, next_obs: Tensor):
        act, next_obs = act.to(self.device), next_obs.to(self.device)
        with self.compute_ctx():
            dist = self.wm.obs_step(self._state[idxes], act, next_obs)
            state = dist.sample()
        state = state.type_as(self._state)
        super().step(idxes, act, state)
        self._state[idxes] = state
