from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import numpy as np
import torch.distributions as D

from .env import Env, EnvSpec
from .spaces import Space
from .spaces.transforms import SpaceTransform, default_cast


class Agent(ABC):
    def __init__(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, obs, info):
        pass

    def observe(self, next_obs, reward, term, trunc, info):
        pass

    @abstractmethod
    def policy(self, cur_obs):
        ...

    def step(self, act):
        pass


class RandomAgent(Agent):
    def __init__(self, env: Env | EnvSpec):
        super().__init__(env.observation_space, env.action_space)

    def policy(self, cur_obs):
        return self.action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        super().__init__(base.observation_space, base.action_space)
        self.base = base

    def reset(self, *args):
        return self.base.reset(*args)

    def observe(self, *args):
        return self.base.observe(*args)

    def policy(self, *args):
        return self.base.policy(*args)

    def step(self, *args):
        return self.base.step(*args)


class WithNoise(AgentWrapper):
    def __init__(self, base: Agent, noise: D.Distribution):
        super().__init__(base)
        self.noise = noise

    def policy(self, *args):
        return self.base.policy(*args) + self.noise.sample()


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        super().__init__(opt.observation_space, opt.action_space)
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self, *args):
        return self._opt.reset(*args), self._rand.reset(*args)

    def observe(self, *args):
        return self._opt.observe(*args), self._rand.observe(*args)

    def policy(self, *args):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.policy(*args)

    def step(self, *args):
        return self._opt.step(*args), self._rand.step(*args)


class WithActionRepeat(AgentWrapper):
    def __init__(self, base: Agent, action_repeat: int):
        super().__init__(base)
        self.action_repeat = action_repeat
        self._policy, self._ctr = None, 0

    def reset(self, *args):
        self._policy, self._ctr = None, 0
        return self.base.reset(*args)

    def policy(self, *args):
        if self._ctr == 0:
            self._policy = self.base.policy(*args)
        return self._policy

    def step(self, *args):
        self._ctr = (self._ctr + 1) % self.action_repeat
        return self.base.step(*args)


class CastAgent(AgentWrapper):
    def __init__(
        self,
        base: Agent,
        observation_map: SpaceTransform | type[Space] = None,
        action_map: SpaceTransform | type[Space] = None,
    ):
        super().__init__(base)

        if observation_map is not None:
            if isinstance(observation_map, type):
                observation_map = default_cast(self.observation_space, observation_map)
            self.observation_space = observation_map.domain
            self._observation_map = observation_map
        else:
            self._observation_map = None

        if action_map is not None:
            if isinstance(action_map, type):
                action_map = default_cast(self.action_space, action_map)
            self.action_space = action_map.codomain
            self._action_map = action_map
            self._inv_action_map = action_map.inv
        else:
            self._action_map = None

    def reset(self, obs, info):
        if self._observation_map is not None:
            obs = self._observation_map(obs)
        return super().reset(obs, info)

    def observe(self, next_obs, reward, term, trunc, info):
        if self._observation_map is not None:
            next_obs = self._observation_map(next_obs)
        return super().observe(next_obs)

    def policy(self, cur_obs):
        act = super().policy(cur_obs)
        if self._action_map is not None:
            act = self._action_map(act)
        return act

    def step(self, act):
        if self._action_map is not None:
            act = self._inv_action_map(act)
        return super().step(act)
