from typing import Protocol
from rsrch.rl.env_spec import EnvSpec
import numpy as np
import torch


class Agent(Protocol):
    def reset(self):
        ...

    def act(self, obs):
        ...


class RandomAgent(Agent):
    def __init__(self, env_spec: EnvSpec):
        self.action_space = env_spec.action_space

    def reset(self):
        ...

    def act(self, obs):
        return self.action_space.sample()


class ToTensor(Agent):
    def __init__(self, base: Agent, device=None):
        self.base = base
        self.device = device

    def reset(self):
        return self.base.reset()

    def act(self, obs):
        action = self.base.act(obs)
        return torch.as_tensor(action, device=self.device)


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self):
        self._opt.reset()
        self._rand.reset()

    def act(self, obs):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.act(obs)


class EpsScheduler:
    def __init__(
        self, agent: EpsAgent, max_eps: float, min_eps: float, step_decay: float
    ):
        self.agent = agent
        self.base_eps = min_eps
        self.eps_amp = max_eps - min_eps
        self.step_decay = step_decay
        self.reset()

        self.cur_eps = max_eps
        agent.eps = self.cur_eps

    def reset(self):
        self._cur_step = 0

    def step(self):
        cur_decay = np.exp(-self.step_decay * self._cur_step)
        self.cur_eps = self.base_eps + self.eps_amp * cur_decay
        self.agent.eps = self.cur_eps
        self._cur_step += 1
