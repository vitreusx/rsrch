from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ._api import *


class Pointwise(VecAgentWrapper):
    """A vec agent wrapper which (implicitly) applies a transform to each "sub-agent" of a vec agent (hence 'pointwise.')"""

    def __init__(self, agent: VecAgent, transform: Callable[[Agent], Agent]):
        super().__init__(agent)
        self.transform = transform
        self._agents: dict[int, Agent] = {}
        self._argv = []
        self._policy = {}

    class Proxy(Agent):
        def __init__(self, parent: "Pointwise", env_idx: int):
            super().__init__()
            self.parent = parent
            self.obs_space = parent.obs_space
            self.act_space = parent.act_space
            self.env_idx = env_idx

        def reset(self, obs):
            self.parent._argv.append((obs,))

        def policy(self):
            return self.parent._policy[self.env_idx]

        def step(self, act, next_obs):
            self.parent._argv.append((act, next_obs))

    def _make_proxy(self, env_idx: int):
        return self.transform(self.Proxy(self, env_idx))

    def reset(self, idxes: np.ndarray, obs_seq):
        self._argv.clear()
        for env_idx, env_obs in zip(idxes, obs_seq):
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            self._agents[env_idx].reset(env_obs)
        self.agent.reset(idxes, *zip(*self._argv))

    def policy(self, idxes: np.ndarray):
        actions = self.agent.policy(idxes)
        for env_idx, action in zip(idxes, actions):
            self._policy[env_idx] = action

        actions = []
        for env_idx in idxes:
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            action = self._agents[env_idx].policy()
            actions.append(action)

        return actions

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self._argv.clear()
        for env_idx, act, next_obs in zip(idxes, act_seq, next_obs_seq):
            if env_idx not in self._agents:
                self._agents[env_idx] = self._make_proxy(env_idx)
            self._agents[env_idx].step(act, next_obs)
        self.agent.step(idxes, *zip(*self._argv))


class RandomAgent(Agent):
    def __init__(self, env: Env):
        super().__init__()
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def policy(self):
        return self.act_space.sample()


class Memoryless(VecAgent, ABC):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self._last_obs = None

    def reset(self, idxes: np.ndarray, obs_seq):
        if self._last_obs is None:
            self._last_obs = obs_seq.clone()
        else:
            self._last_obs[idxes] = obs_seq

    def policy(self, idxes: np.ndarray):
        return self._policy(self._last_obs[idxes])

    @abstractmethod
    def _policy(self, last_obs):
        raise NotImplementedError()

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self._last_obs[idxes] = next_obs_seq


class RandomVecAgent(Memoryless):
    def __init__(self, envs: VecEnv):
        super().__init__(envs.obs_space, envs.act_space)

    def _policy(self, last_obs):
        return self.act_space.sample((last_obs.shape[0],))
