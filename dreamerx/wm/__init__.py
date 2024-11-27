from contextlib import contextmanager
from functools import cached_property
from typing import Any, Callable

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym

from ..common.utils import autocast


class WorldModel(nn.Module):
    obs_space: Any
    act_space: Any
    state_space: Any

    reward_dec: Callable[[Tensor], D.Distribution]
    term_dec: Callable[[Tensor], D.Distribution]

    def reset(
        self,
        obs: Tensor,
    ) -> D.Distribution: ...

    def obs_step(
        self,
        state: Tensor,
        act: Tensor,
        next_obs: Tensor,
    ) -> D.Distribution: ...

    def img_step(
        self,
        state: Tensor,
        act: Tensor,
    ) -> D.Distribution: ...

    def observe(
        self,
        input: tuple[Tensor, Tensor],
        h_0: list[Tensor | None],
    ) -> tuple[Any, Tensor]: ...

    def imagine(
        self,
        act_seq: Tensor,
        h_0: list[Tensor | None],
    ) -> tuple[Any, Tensor]: ...


class Agent(gym.VecAgentWrapper):
    def __init__(
        self,
        agent: gym.VecAgent,
        wm: WorldModel,
        pass_tensors: bool = True,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(agent)
        self.wm = wm
        self.obs_space = self.wm.obs_space
        self.act_space = self.wm.act_space
        self.pass_tensors = pass_tensors
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
        super().reset(idxes, self._cast_state(state))
        if self._state is None:
            self._state = state.clone()
        else:
            self._state[idxes] = state.type_as(self._state)

    def policy(self, idxes):
        return super().policy(idxes)

    def step(self, idxes, act: Tensor, next_obs: Tensor):
        act, next_obs = act.to(self.device), next_obs.to(self.device)
        with self.compute_ctx():
            dist = self.wm.obs_step(self._state[idxes], act, next_obs)
            state = dist.sample()
        state = state.type_as(self._state)
        super().step(idxes, act, self._cast_state(state))
        self._state[idxes] = state

    def _cast_state(self, state):
        if self.pass_tensors and not isinstance(state, Tensor):
            state = state.as_tensor()
        return state
