from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import numpy as np
import torch.distributions as D

from .env import Env, EnvSpec
from .spaces import Space
from .spaces.transforms import SpaceTransform, default_cast


class Agent(ABC, Generic[ObsType, ActType]):
    def __init__(self, observation_space: Space, action_space: Space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, obs: ObsType):
        return self.observe(obs)

    def observe(self, obs: ObsType):
        pass

    @abstractmethod
    def policy(self, obs=None) -> ActType:
        ...

    def step(self, act: ActType):
        pass


class RandomAgent(Agent):
    def __init__(self, env: Env | EnvSpec):
        super().__init__(env.observation_space, env.action_space)

    def policy(self, obs=None):
        return self.action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        super().__init__(base.observation_space, base.action_space)
        self.base = base

    def reset(self, obs):
        return self.base.reset(obs)

    def observe(self, obs):
        return self.base.observe(obs)

    def policy(self, obs=None):
        return self.base.policy(obs)

    def step(self, act):
        return self.base.step(act)


class WithNoise(AgentWrapper):
    def __init__(self, base: Agent, noise: D.Distribution):
        super().__init__(base)
        self.noise = noise

    def policy(self, obs=None):
        return self.base.policy(obs) + self.noise.sample()


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        super().__init__(opt.observation_space, opt.action_space)
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self, obs):
        return self._opt.reset(obs), self._rand.reset(obs)

    def observe(self, obs):
        return self._opt.observe(obs), self._rand.observe(obs)

    def policy(self, obs=None):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.policy(obs)

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)


class WithActionRepeat(AgentWrapper):
    def __init__(self, base: Agent, action_repeat: int):
        super().__init__(base)
        self.action_repeat = action_repeat
        self._policy, self._ctr = None, 0

    def reset(self, obs):
        self._policy, self._ctr = None, 0
        return self.base.reset(obs)

    def policy(self, obs=None):
        if self._ctr == 0:
            self._policy = self.base.policy(obs)
        return self._policy

    def step(self, act):
        self._ctr = (self._ctr + 1) % self.action_repeat
        return self.base.step(act)


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

    def reset(self, obs):
        if self._observation_map is not None:
            obs = self._observation_map(obs)
        return super().reset(obs)

    def observe(self, obs):
        if self._observation_map is not None:
            obs = self._observation_map(obs)
        return super().observe(obs)

    def policy(self, obs=None):
        act = super().policy(obs)
        if self._action_map is not None:
            act = self._action_map(act)
        return act

    def step(self, act):
        if self._action_map is not None:
            act = self._inv_action_map(act)
        return super().step(act)
