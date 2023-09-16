from abc import ABC, abstractmethod

import numpy as np
from .base import VectorEnv


class Agent(ABC):
    def reset(self, idxes, obs, info):
        pass

    @abstractmethod
    def policy(self, obs):
        ...

    def step(self, act):
        pass

    def observe(self, idxes, next_obs, term, trunc, info):
        pass


class RandomAgent(Agent):
    def __init__(self, env: VectorEnv):
        super().__init__()
        self._action_space = env.action_space

    def policy(self, _):
        return self._action_space.sample()


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float, num_envs: int):
        super().__init__()
        self._opt = opt
        self._rand = rand
        self.eps = eps
        self.num_envs = num_envs

    def reset(self, *args, **kwargs):
        self._opt.reset(*args, **kwargs)
        self._rand.reset(*args, **kwargs)

    def observe(self, *args, **kwargs):
        self._opt.observe(*args, **kwargs)
        self._rand.observe(*args, **kwargs)

    def policy(self, obs):
        use_rand = np.random.rand(self.num_envs) < self.eps
        opt_p, rand_p = self._opt.policy(obs), self._rand.policy(obs)
        return np.where(use_rand, rand_p, opt_p)

    def step(self, act):
        self._opt.step(act)
        self._rand.step(act)


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        super().__init__()
        self._agent = agent

    def reset(self, *args, **kwargs):
        return self._agent.reset(*args, **kwargs)

    def policy(self, obs):
        return self._agent.policy(obs)

    def step(self, act):
        return self._agent.step(act)

    def observe(self, *args, **kwargs):
        return self._agent.observe(*args, **kwargs)
