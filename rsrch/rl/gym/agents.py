from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor

from . import spaces
from .env import Env, EnvSpec
from .spaces import Space


class Agent(ABC, Generic[ObsType, ActType]):
    obs_space: Space
    act_space: Space

    def reset(self, obs=None):
        if obs is not None:
            self.observe(obs)

    def observe(self, obs: ObsType):
        pass

    @abstractmethod
    def policy(self) -> ActType:
        ...

    def step(self, act: ActType):
        pass


class RandomAgent(Agent):
    def __init__(self, env_spec: EnvSpec):
        self.action_space = env_spec.action_space

    def policy(self):
        return self.action_space.sample()


class AgentWrapper(Agent):
    def __init__(self, base: Agent):
        self.base = base
        self.obs_space = base.obs_space
        self.act_space = base.act_space

    def reset(self, obs):
        return self.base.reset(obs)

    def observe(self, obs):
        return self.base.observe(obs)

    def policy(self):
        return self.base.policy()

    def step(self, act):
        return self.base.step(act)


class LambdaWrap(AgentWrapper):
    def __init__(
        self,
        base: Agent,
        obs_fn=None,
        pi_fn=None,
        step_fn=None,
        obs_space=None,
        act_space=None,
    ):
        self.base = base
        self.obs_space = obs_space if obs_space is not None else self.base.obs_space
        self.act_space = act_space if act_space is not None else self.base.act_space
        self.obs_fn = obs_fn
        self.pi_fn = pi_fn
        self.step_fn = step_fn

    def reset(self, obs):
        if self.obs_fn is not None:
            obs = self.obs_fn(obs)
        return self.base.reset(obs)

    def observe(self, obs):
        if self.obs_fn is not None:
            obs = self.obs_fn(obs)
        return self.base.observe(obs)

    def policy(self):
        policy = self.base.policy()
        if self.pi_fn is not None:
            policy = self.pi_fn(policy)
        return policy

    def step(self, act):
        if self.step_fn is not None:
            act = self.step_fn(act)
        return self.base.step(act)


class WithNoise(AgentWrapper):
    def __init__(self, base: Agent, noise_fn: Callable[[], D.Distribution]):
        super().__init__(base)
        self.noise_fn = noise_fn

    def reset(self, obs):
        self.noise = self.noise_fn()
        return self.base.reset(obs)

    def policy(self):
        return self.base.policy() + self.noise.sample()


class AdaptToEnv(AgentWrapper):
    def __init__(self, base: Agent, target_env: Env):
        super().__init__(base)
        self.spec = EnvSpec(target_env)
        self.obs_space = self.spec.observation_space
        self.act_space = self.spec.action_space

    def _conv_obs(self, obs, from_space: Space, to_space: Space):
        if isinstance(from_space, spaces.Box):
            if isinstance(to_space, spaces.TensorBox):
                dtype = to_space.dtype
                device = to_space.device
                obs = torch.as_tensor(obs, dtype=dtype, device=device)
        elif isinstance(from_space, spaces.TensorBox):
            if isinstance(to_space, spaces.Box):
                obs = obs.detach().cpu().numpy()
        return obs

    def reset(self, obs):
        obs = self._conv_obs(obs, self.obs_space, self.base.obs_space)
        return self.base.reset(obs)

    def observe(self, obs):
        obs = self._conv_obs(obs, self.obs_space, self.base.obs_space)
        return self.base.observe(obs)

    def _conv_act(self, act, from_space: Space, to_space: Space):
        if isinstance(from_space, spaces.Box):
            if isinstance(to_space, spaces.TensorBox):
                dtype = to_space.dtype
                device = to_space.device
                obs = torch.as_tensor(obs, dtype=dtype, device=device)
            elif isinstance(to_space, spaces.Discrete):
                

    def policy(self):
        return self.base.policy()

    def step(self, act):
        return self.base.step(act)


class ToTensor(AgentWrapper):
    def __init__(self, base: Agent, device=None):
        super().__init__(base)
        self.device = device

    def observe(self, obs):
        return self.base.observe(obs.detach().cpu().numpy())

    def policy(self) -> Tensor:
        policy = self.base.policy()
        if isinstance(policy, Tensor):
            policy = policy.to(self.device)
        else:
            policy = torch.as_tensor(np.asarray(policy), device=self.device)
        return policy

    def step(self, act):
        return self.base.step(act.detach().cpu().numpy())


class FromTensor(AgentWrapper):
    def __init__(self, base: Agent, env_spec: EnvSpec, device=None):
        super().__init__(base)
        self.device = device
        self.env_spec = env_spec

    def observe(self, obs):
        if isinstance(obs, Tensor):
            obs = obs.to(self.device)
        else:
            obs = torch.as_tensor(np.asarray(obs), device=self.device)
        return self.base.observe(obs)

    def policy(self):
        space = self.env_spec.action_space
        if isinstance(space, (spaces.TensorBox, spaces.TensorDiscrete)):
            return self.base.policy()
        elif isinstance(space, spaces.Box):
            return self.base.policy().detach().cpu().numpy()
        elif isinstance(space, spaces.Discrete):
            return self.base.policy().item()

    def step(self, act):
        if isinstance(act, Tensor):
            act = act.to(self.device)
        else:
            act = torch.as_tensor(np.asarray(act), device=self.device)
        return self.base.step(act)


class EpsAgent(Agent):
    def __init__(self, opt: Agent, rand: Agent, eps: float):
        self._opt = opt
        self._rand = rand
        self.eps = eps

    def reset(self, obs):
        return self._opt.reset(obs), self._rand.reset(obs)

    def observe(self, obs):
        return self._opt.observe(obs), self._rand.observe(obs)

    def policy(self):
        use_rand = np.random.rand() < self.eps
        agent = self._rand if use_rand else self._opt
        return agent.policy()

    def step(self, act):
        return self._opt.step(act), self._rand.step(act)


class WithActionRepeat(AgentWrapper):
    def __init__(self, base: Agent, action_repeat: int):
        super().__init__(base)
        self.action_repeat = action_repeat
        self._policy, self._ctr = None, 1

    def reset(self, obs):
        self._policy, self._ctr = None, 0
        return self.base.reset(obs)

    def policy(self):
        if self._policy is None or self._ctr == 0:
            self._policy = self.base.policy()
        return self._policy

    def step(self, act):
        self._ctr = (self._ctr + 1) % self.action_repeat
        return self.base.step(act)
