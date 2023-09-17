from abc import ABC, abstractmethod

import numpy as np
import torch.distributions as D

from .envs import Env


class Agent(ABC):
    def reset(self, obs, info):
        pass

    def observe(self, act, next_obs, reward, term, trunc, info):
        pass

    @abstractmethod
    def policy(self, obs):
        ...

    def step(self, act):
        pass


class RandomAgent(Agent):
    def __init__(self, env: Env):
        super().__init__()
        self._action_space = env.action_space

    def policy(self, _):
        return self._action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        super().__init__()
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
        super().__init__()
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
