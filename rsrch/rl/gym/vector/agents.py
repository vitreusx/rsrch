from abc import ABC, abstractmethod

import numpy as np

from .base import utils
from ..env import EnvSpec, VectorEnv
from .. import spaces
from ..spaces import Space


class VecAgent(ABC):
    observation_space: Space
    single_observation_space: Space
    action_space: Space
    single_action_space: Space

    def __init__(self, num_envs: int, observation_space: Space, action_space: Space):
        self.num_envs = num_envs
        self.single_observation_space = observation_space
        self.observation_space = utils.batch_space(observation_space, num_envs)
        self.single_action_space = action_space
        self.action_space = utils.batch_space(action_space, num_envs)

    def reset(self, obs, mask):
        return self.observe(obs, mask)

    def observe(self, obs, mask):
        pass

    @abstractmethod
    def policy(self, obs=None):
        ...

    def step(self, act):
        pass


class RandomVecAgent(VecAgent):
    def __init__(self, env: VectorEnv | EnvSpec):
        super().__init__(
            env.num_envs,
            env.single_observation_space,
            env.single_action_space,
        )

    def policy(self, obs=None):
        return self.action_space.sample()


class EpsVecAgent(VecAgent):
    def __init__(self, opt: VecAgent, rand: VecAgent, eps: float, num_envs: int):
        super().__init__(
            num_envs,
            opt.single_action_space,
            opt.single_observation_space,
        )
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self, obs, mask):
        return self._opt.reset(obs, mask), self._rand.reset(obs, mask)

    def observe(self, obs, mask):
        return self._opt.observe(obs, mask), self._rand.observe(obs, mask)

    def policy(self, obs=None):
        use_rand = np.random.rand(self.num_envs) < self.eps
        opt_p, rand_p = self._opt.policy(obs), self._rand.policy(obs)
        return [rand_p[i] if use_rand[i] else opt_p[i] for i in range(self.num_envs)]

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)


class VecAgentWrapper(VecAgent):
    def __init__(self, agent: VecAgent):
        super().__init__(
            agent.num_envs,
            agent.single_observation_space,
            agent.single_action_space,
        )
        self._agent = agent

    def reset(self, obs, mask):
        return self._agent.reset(obs, mask)

    def observe(self, obs, mask):
        return self._agent.observe(obs, mask)

    def policy(self, obs=None):
        return self._agent.policy(obs)

    def step(self, act):
        return self._agent.step(act)
