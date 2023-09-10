from abc import ABC, abstractmethod

import numpy as np
from ..env import EnvSpec, VectorEnv
from .events import *


class Agent(ABC):
    def reset(self, data: VecReset):
        pass

    @abstractmethod
    def policy(self, obs):
        ...

    def step(self, act):
        pass

    def observe(self, data: VecStep):
        pass


class RandomAgent(Agent):
    def __init__(self, env: VectorEnv | EnvSpec):
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

    def reset(self, *args):
        return self._opt.reset(*args), self._rand.reset(*args)

    def observe(self, *args):
        return self._opt.observe(*args), self._rand.observe(*args)

    def policy(self, last_obs):
        use_rand = np.random.rand(self.num_envs) < self.eps
        opt_p, rand_p = self._opt.policy(last_obs), self._rand.policy(last_obs)
        return [rand_p[i] if use_rand[i] else opt_p[i] for i in range(self.num_envs)]

    def step(self, *args):
        return self._opt.step(*args), self._rand.step(*args)


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        super().__init__()
        self._agent = agent

    def reset(self, data: VecReset):
        return self._agent.reset(data)

    def policy(self, obs):
        return self._agent.policy(obs)

    def step(self, act):
        return self._agent.step(act)

    def observe(self, data: VecStep):
        return self._agent.observe(data)
