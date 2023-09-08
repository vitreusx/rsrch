from abc import ABC, abstractmethod

import numpy as np

from . import utils
from ..env import EnvSpec, VectorEnv
from ..spaces import Space


class VecAgent(ABC):
    observation_space: Space
    single_observation_space: Space
    action_space: Space
    single_action_space: Space

    def __init__(self, num_envs: int, observation_space: Space, action_space: Space):
        self.num_envs = num_envs
        # self.single_observation_space = observation_space
        # self.observation_space = utils.batch_space(observation_space, num_envs)
        # self.single_action_space = action_space
        # self.action_space = utils.batch_space(action_space, num_envs)

    def reset(self, env_idx, obs, info):
        pass

    @abstractmethod
    def policy(self, last_obs):
        ...

    def step(self, act):
        pass

    def observe(self, env_idx, obs, reward, term, trunc, info):
        pass


class RandomVecAgent(VecAgent):
    def __init__(self, env: VectorEnv | EnvSpec):
        super().__init__(
            env.num_envs,
            env.single_observation_space,
            env.single_action_space,
        )

    def policy(self, _):
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


class VecAgentWrapper(VecAgent):
    def __init__(self, agent: VecAgent):
        super().__init__(
            agent.num_envs,
            agent.single_observation_space,
            agent.single_action_space,
        )
        self._agent = agent

    def reset(self, *args):
        return self._agent.reset(*args)

    def observe(self, *args):
        return self._agent.observe(*args)

    def policy(self, last_obs):
        return self._agent.policy(last_obs)

    def step(self, *args):
        return self._agent.step(*args)
