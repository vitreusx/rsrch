from typing import Callable, Protocol

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor

from rsrch.rl.api import Agent
from rsrch.rl.spec import EnvSpec


class RandomAgent(Agent):
    def __init__(self, env_spec: EnvSpec):
        self.action_space = env_spec.action_space

    def policy(self):
        return self.action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        self.base = base

    def reset(self):
        return self.base.reset()

    def observe(self, obs):
        return self.base.observe(obs)

    def policy(self):
        return self.base.policy()

    def step(self, act):
        return self.base.step(act)


class WithNoise(AgentWrapper):
    def __init__(self, base: Agent, noise_fn: Callable[[], D.Distribution]):
        super().__init__(base)
        self.noise_fn = noise_fn

    def reset(self):
        self.noise = self.noise_fn()
        return self.base.reset()

    def policy(self):
        return self.base.policy() + self.noise.sample()


class ToTensor(AgentWrapper):
    def __init__(self, base: Agent, device=None):
        super().__init__(base)
        self.device = device

    def observe(self, obs):
        return self.base.observe(obs.detach().cpu().numpy())

    def policy(self) -> Tensor:
        return torch.as_tensor(self.base.policy(), device=self.device)

    def step(self, act):
        return self.base.step(act.detach().cpu().numpy())


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self):
        return self._opt.reset(), self._rand.reset()

    def observe(self, obs):
        return self._opt.observe(obs), self._rand.observe(obs)

    def policy(self):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.policy()

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)


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


class WithActionRepeat(AgentWrapper):
    def __init__(self, base: Agent, action_repeat: int):
        super().__init__(base)
        self.action_repeat = action_repeat
        self._policy, self._ctr = None, 1

    def reset(self):
        self._policy, self._ctr = None, 0
        return self.base.reset()

    def policy(self):
        if self._policy is None or self._ctr == 0:
            self._policy = self.base.policy()
        return self._policy

    def step(self, act):
        self._ctr = (self._ctr + 1) % self.action_repeat
        return self.base.step(act)
