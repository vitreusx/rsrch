from typing import Any, List, Optional, Union

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym

from .. import api


class WorldModel(nn.Module, api.WorldModel):
    obs_space: gym.Space
    act_space: gym.Space

    def __init__(self):
        super().__init__()
        self._dev: Tensor
        self.register_buffer("_dev", torch.empty([]))

    @property
    def device(self):
        return self._dev.device

    def reset(self, batch_size: int) -> Tensor:
        ...

    def step(self, s: Tensor, a: Tensor) -> D.Distribution:
        ...

    def rew(self, next_s: Tensor) -> Tensor:
        ...

    def term(self, s: Tensor) -> Tensor:
        ...

    def act_enc(self, a) -> Tensor:
        ...

    def act_dec(self, enc_a: Tensor) -> Any:
        ...


class Env(gym.VectorEnv):
    def __init__(self, wm: WorldModel, num_envs: int):
        super().__init__(num_envs, wm.obs_space, wm.act_space)
        self.wm = wm

    def reset_wait(self, *args, **kwargs):
        self._state = self.wm.reset(self.num_envs)
        return self._state, {}

    def step_async(self, actions):
        a = self.wm.act_enc(actions)
        self._state = self.wm.step(self._state, a).rsample()
        rew = self.wm.rew(self._state)
        term = self.wm.term(self._state)
        return self._state, rew, term, False, {}
