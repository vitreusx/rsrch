from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from ..api import Agent, VecAgent, VecAgentWrapper, VecEnv


class Pointwise(VecAgentWrapper):
    """A vec agent wrapper which (implicitly) applies a transform to each
    "sub-agent" of a vec agent (hence 'pointwise.')"""

    def __init__(self, agent: VecAgent, transform: Callable[[Agent], Agent]):
        super().__init__(agent)
        self.transform = transform
        self._agents: dict[int, Agent] = {}
        self._argv = []
        self._policy = {}

    class Proxy(Agent):
        def __init__(self, parent: "Pointwise", env_idx: int):
            super().__init__(parent.obs_space, parent.act_space)
            self.parent = parent
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


class Markov(VecAgent, ABC):
    """A helper class for implementing 'Markovian' vector agents (that is,
    agents which decide based on the last observation alone.)"""

    def __init__(self, obs_space, act_space):
        super().__init__(obs_space, act_space)
        self._last_obs = None

    def reset(self, idxes: np.ndarray, obs_seq):
        if self._last_obs is None:
            self._last_obs = obs_seq.clone()
        else:
            self._last_obs[idxes] = obs_seq

    def policy(self, idxes: np.ndarray):
        return self.get_policy(self._last_obs[idxes])

    @abstractmethod
    def get_policy(self, last_obs):
        raise NotImplementedError()

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self._last_obs[idxes] = next_obs_seq


class RandomVecAgent(VecAgent):
    def __init__(
        self,
        envs: VecEnv | None = None,
        obs_space: Any | None = None,
        act_space: Any | None = None,
    ):
        if obs_space is None:
            obs_space = envs.obs_space
        if act_space is None:
            act_space = envs.act_space
        super().__init__(obs_space, act_space)

    def policy(self, idxes: np.ndarray):
        return self.act_space.sample((len(idxes),))
