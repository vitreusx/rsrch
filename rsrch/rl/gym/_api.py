from abc import ABC, abstractmethod
from typing import Any, Iterable

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


class Env(ABC):
    obs_space: Any
    act_space: Any

    @abstractmethod
    def reset(self) -> dict:
        ...

    @abstractmethod
    def step(self, act) -> tuple[dict, bool]:
        ...

    def rollout(self, agent: Agent):
        obs = None
        while True:
            if obs is None:
                obs = self.reset()
                agent.reset(obs)
                yield obs, False

            act = agent.policy()
            next_obs, final = self.step(act)
            agent.step(act, next_obs)
            yield {**next_obs, "act": act}, final

            obs = next_obs
            if final:
                obs = None


class EnvWrapper(Env):
    def __init__(self, env: Env):
        super().__init__()
        self.env = env
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def reset(self):
        return self.env.reset()

    def step(self, act):
        return self.env.step(act)


class VecEnv(ABC):
    num_envs: int
    obs_space: Any
    act_space: Any

    @abstractmethod
    def rollout(self, agent: VecAgent) -> Iterable[tuple[int, tuple[dict, bool]]]:
        ...


class VecEnvWrapper(VecEnv):
    def __init__(self, env: VecEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.act_space = env.act_space
        self.obs_space = env.obs_space

    def rollout(self, agent):
        return self.env.rollout(agent)
