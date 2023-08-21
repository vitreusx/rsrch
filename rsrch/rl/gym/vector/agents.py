from abc import ABC, abstractmethod

import numpy as np

from ..env import EnvSpec, VectorEnv


class VecAgent(ABC):
    def reset(self, obs, mask):
        return self.observe(obs, mask)

    def observe(self, obs, mask):
        pass

    @abstractmethod
    def policy(self):
        ...

    def step(self, act):
        pass


class RandomVecAgent(VecAgent):
    def __init__(self, env: VectorEnv | EnvSpec):
        env_spec = EnvSpec(env)
        self.action_space = env_spec.action_space

    def policy(self):
        return self.action_space.sample()


class EpsVecAgent(VecAgent):
    def __init__(self, opt: VecAgent, rand: VecAgent, eps: float, num_envs: int):
        self._opt = opt
        self._rand = rand
        self.eps = eps
        self.num_envs = num_envs

    def reset(self, obs, mask):
        return self._opt.reset(obs, mask), self._rand.reset(obs, mask)

    def observe(self, obs, mask):
        return self._opt.observe(obs, mask), self._rand.observe(obs, mask)

    def policy(self):
        use_rand = np.random.rand(self.num_envs) < self.eps
        opt_p, rand_p = self._opt.policy(), self._rand.policy()
        return [rand_p[i] if use_rand[i] else opt_p[i] for i in range(self.num_envs)]

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)
