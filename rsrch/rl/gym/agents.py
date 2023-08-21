from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import numpy as np
import torch.distributions as D

from .env import Env, EnvSpec


class Agent(ABC, Generic[ObsType, ActType]):
    def reset(self, obs: ObsType):
        return self.observe(obs)

    def observe(self, obs: ObsType):
        pass

    @abstractmethod
    def policy(self) -> ActType:
        ...

    def step(self, act: ActType):
        pass


class RandomAgent(Agent):
    def __init__(self, env: Env | EnvSpec):
        env_spec = EnvSpec(env)
        self.action_space = env_spec.action_space

    def policy(self):
        return self.action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        self.base = base
        self.obs_space = base.obs_space
        self.act_space = base.act_space

    def reset(self, obs):
        return self.base.reset(obs)

    def observe(self, obs):
        return self.base.observe(obs)

    def policy(self):
        return self.base.policy()

    def step(self, act):
        return self.base.step(act)


class WithNoise(AgentWrapper):
    def __init__(self, base: Agent, noise_fn: Callable[[], D.Distribution]):
        super().__init__(base)
        self.noise_fn = noise_fn

    def reset(self, obs):
        self.noise = self.noise_fn()
        return self.base.reset(obs)

    def policy(self):
        return self.base.policy() + self.noise.sample()


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self, obs):
        return self._opt.reset(obs), self._rand.reset(obs)

    def observe(self, obs):
        return self._opt.observe(obs), self._rand.observe(obs)

    def policy(self):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.policy()

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)


class WithActionRepeat(AgentWrapper):
    def __init__(self, base: Agent, action_repeat: int):
        super().__init__(base)
        self.action_repeat = action_repeat
        self._policy, self._ctr = None, 1

    def reset(self, obs):
        self._policy, self._ctr = None, 0
        return self.base.reset(obs)

    def policy(self):
        if self._policy is None or self._ctr == 0:
            self._policy = self.base.policy()
        return self._policy

    def step(self, act):
        self._ctr = (self._ctr + 1) % self.action_repeat
        return self.base.step(act)
