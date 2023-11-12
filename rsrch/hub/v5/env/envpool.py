from copy import deepcopy

import envpool
import numpy as np

import rsrch.rl.gym.vector.utils as vec_utils
from rsrch.rl import gym
from rsrch.rl.gym.spaces.tensor import *


class VecEnvPool(gym.VectorEnv):
    """A proper adapter for envpool environments."""

    def __init__(self, envp: gym.Env):
        super().__init__(
            len(envp.all_env_ids),
            envp.observation_space,
            envp.action_space,
        )
        self._envp = envp

    @staticmethod
    def make(task_id: str, env_type: str, **kwargs):
        envp = envpool.make(task_id, env_type, **kwargs)
        env = VecEnvPool(envp)
        return env

    def reset_async(self, seed=None, options=None):
        pass

    def reset_wait(self, seed=None, options=None):
        obs, info = self._envp.reset()
        return obs, {}

    def step_async(self, actions):
        if isinstance(self.single_action_space, gym.spaces.Discrete):
            actions = actions.astype(np.int32)
        self._actions = actions

    def step_wait(self, **kwargs):
        next_obs, reward, term, trunc, info = self._envp.step(self._actions)
        info = {}
        done: np.ndarray = term | trunc
        if done.any():
            info["_final_observation"] = done.copy()
            info["_final_info"] = done.copy()
            info["final_observation"] = np.array([None for _ in range(self.num_envs)])
            info["final_info"] = np.array([None for _ in range(self.num_envs)])

            for i in range(self.num_envs):
                if done[i]:
                    info["final_observation"][i] = next_obs[i].copy()
                    info["final_info"][i] = {}

            reset_ids = self._envp.all_env_ids[done]
            reset_obs, reset_info = self._envp.reset(reset_ids)
            next_obs[reset_ids] = reset_obs

        return next_obs, reward, term, trunc, info

    def __repr__(self):
        return f"VecEnvPool({self._envp!r})"

    def __str__(self):
        return f"VecEnvPool({self._envp})"

class BoxTransform(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, f):
        super().__init__(env)
        self._f = f

        assert isinstance(env.observation_space, gym.spaces.Box)
        new_shape = self.observation(env.observation_space.sample()).shape
        new_low = self.observation(env.observation_space.low)
        new_high = self.observation(env.observation_space.high)
        new_dtype = env.observation_space.dtype
        new_seed = deepcopy(env.observation_space.np_random)
        new_space = gym.spaces.Box(new_low, new_high, new_shape, new_dtype, new_seed)
        self.observation_space = new_space

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self._f(obs)


class ObservationWrapper(gym.vector.VectorEnvWrapper):
    def __init__(self, env: gym.VectorEnv):
        super().__init__(env)
        self._prev_space = env.single_observation_space

    def observation(self, obs):
        raise NotImplementedError()

    def reset_wait(self, **kwargs):
        obs, info = super().reset_wait(**kwargs)
        obs = self.observation(obs)
        return obs, info

    def step_wait(self):
        next_obs, reward, term, trunc, info = super().step_wait()
        next_obs = self.observation(next_obs)
        if "final_observation" in info:
            mask, final_obs = info["_final_observation"], info["final_observation"]
            idxes = np.where(mask)[0]
            obs_space = self.single_observation_space
            t = vec_utils.stack(self._prev_space, [*final_obs[idxes]])
            t = vec_utils.split(obs_space, self.observation(t), len(idxes))
            for i, env_i in enumerate(idxes):
                info["final_observation"][env_i] = t[i]
        return next_obs, reward, term, trunc, info


class BoxTransformV(ObservationWrapper):
    def __init__(self, env: gym.VectorEnv, f):
        super().__init__(env)
        self.observation = f

        assert isinstance(env.single_observation_space, gym.spaces.Box)
        obs_space = env.single_observation_space
        dummy_x = self.observation(obs_space.sample()[None])[0]
        new_shape = dummy_x.shape
        new_dtype = dummy_x.dtype
        new_low = self.observation(obs_space.low[None])[0]
        new_high = self.observation(obs_space.high[None])[0]
        new_seed = deepcopy(obs_space.np_random)
        new_space = gym.spaces.Box(new_low, new_high, new_shape, new_dtype, new_seed)
        self.single_observation_space = new_space
        self.observation_space = vec_utils.batch_space(
            self.single_observation_space, self.num_envs
        )


def np_flatten(x: np.ndarray, start_axis=0, end_axis=-1):
    shape = x.shape
    start_axis = range(len(shape))[start_axis]
    end_axis = range(len(shape))[end_axis]
    flat_n = int(np.prod(x.shape[start_axis : end_axis + 1]))
    new_shape = [*x.shape[:start_axis], flat_n, *x.shape[end_axis + 1 :]]
    return x.reshape(new_shape)
