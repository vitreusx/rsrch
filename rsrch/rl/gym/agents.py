from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Agent(ABC):
    def reset(self, obs):
        pass

    @abstractmethod
    def policy(self):
        ...

    def step(self, act, next_obs):
        pass


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def reset(self, step):
        return self.agent.reset(step)


class VecAgent(ABC):
    def reset(self, idxes: np.ndarray, obs_seq):
        pass

    @abstractmethod
    def policy(self, idxes: np.ndarray):
        ...

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        pass


class VecAgentWrapper(VecAgent):
    def __init__(self, agent: VecAgent):
        super().__init__()
        self.agent = agent

    def reset(self, idxes: np.ndarray, obs_seq):
        self.agent.reset(idxes, obs_seq)

    def policy(self, idxes: np.ndarray):
        return self.agent.policy(idxes)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self.agent.step(idxes, act_seq, next_obs_seq)


class Pointwise(VecAgent):
    """A vec agent wrapper which (implicitly) applies a transform to each "sub-agent" of a vec agent (hence 'pointwise.')"""

    def __init__(self, agent: VecAgent, transform: Callable[[Agent], Agent]):
        super().__init__()
        self.agent = agent
        self.transform = transform
        self._agents: dict[int, Agent] = {}
        self._argv = []
        self._policy = {}

    class Proxy(Agent):
        def __init__(self, parent: "Pointwise", env_idx: int):
            super().__init__()
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
