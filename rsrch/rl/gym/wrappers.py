from gymnasium import Wrapper
from gymnasium.wrappers import *
import torch
from .spaces import *
from .spaces import transforms as T
from .spaces.transforms import SpaceTransform, default_cast
from .env import Env


class KeepState(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def step(self, act):
        result = super().step(act)
        self._obs = result[0]
        return result


class CastEnv(Wrapper):
    def __init__(
        self,
        env: Env,
        observation_map: SpaceTransform | type[Space] = None,
        action_map: SpaceTransform | type[Space] = None,
    ):
        super().__init__(env)

        if observation_map is not None:
            if isinstance(observation_map, type):
                observation_map = default_cast(self.observation_space, observation_map)
            self.observation_space = observation_map.codomain
            self._observation_map = observation_map
        else:
            self._observation_map = None

        if action_map is not None:
            if isinstance(action_map, type):
                action_map = default_cast(self.action_space, action_map)
            self.action_space = action_map.domain
            self._action_map = action_map
        else:
            self._action_map = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._observation_map is not None:
            obs = self._observation_map(obs)
        return obs, info

    def step(self, action):
        if self._action_map is not None:
            action = self._action_map(action)
        next_obs, reward, term, trunc, info = super().step(action)
        if self._observation_map is not None:
            next_obs = self._observation_map(next_obs)
        return next_obs, reward, term, trunc, info


class ToTensor(CastEnv):
    def __init__(self, env: Env, device=None):
        super().__init__(
            env=env,
            observation_map=T.ToTensor(env.observation_space, device),
            action_map=T.ToTensor(env.action_space, device),
        )


class ConcatObs(Wrapper):
    def __init__(self, env: Env, num_stack: int):
        super().__init__(env)
        self._num_stack = num_stack

        obs_space = self.observation_space
        shape = [
            num_stack * dim if idx == 0 else dim
            for idx, dim in enumerate(obs_space.shape)
        ]
        shape = torch.Size(shape)
        if isinstance(obs_space, TensorImage):
            obs_space = TensorImage(shape, obs_space._normalized, obs_space.device)
        elif isinstance(obs_space, TensorBox):
            low = obs_space.low.expand(shape)
            high = obs_space.high.expand(shape)
            obs_space = TensorBox(low, high, shape, obs_space.device, obs_space.dtype)
        else:
            raise ValueError(type(obs_space))
        self.observation_space = obs_space

        self._obs: torch.Tensor = self.observation_space.sample()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._obs[:] = obs.expand(self.observation_space.shape)
        return self._obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = super().step(action)
        self._obs[:-1] = self._obs[1:].clone()
        self._obs[-1] = next_obs
        return self._obs, reward, term, trunc, info
