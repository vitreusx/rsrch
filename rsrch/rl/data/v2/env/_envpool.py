import envpool
import numpy as np

from rsrch.rl import gym
from rsrch.rl.gym.wrappers import LazyFrames


class VecEnvPool(gym.VectorEnv):
    """A proper adapter for envpool environments."""

    def __init__(self, envp: gym.Env, num_stack: int | None):
        self.num_stack = num_stack

        obs_space = envp.observation_space

        if num_stack is not None:
            assert type(obs_space) == gym.spaces.Box
            low = self._unstack(obs_space.low)
            high = self._unstack(obs_space.high)
            obs_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=low.shape,
                dtype=obs_space.dtype,
                seed=obs_space._np_random,
            )

        super().__init__(
            len(envp.all_env_ids),
            obs_space,
            envp.action_space,
        )
        self._envp = envp

    def _unstack(self, obs):
        return obs.reshape([len(obs), self.num_stack, -1, *obs.shape[2:]])

    @staticmethod
    def make(task_id: str, env_type: str, **kwargs):
        envp = envpool.make(task_id, env_type, **kwargs)
        stack_num = kwargs.get("stack_num", None)
        env = VecEnvPool(envp, num_stack=stack_num)
        return env

    def reset_async(self, seed=None, options=None):
        pass

    def reset_wait(self, seed=None, options=None):
        obs, info = self._envp.reset()
        if self.num_stack is not None:
            obs = self._unstack(obs)
        return obs, {}

    def step_async(self, actions):
        if isinstance(self.single_action_space, gym.spaces.Discrete):
            actions = actions.astype(np.int32)
        self._envp.send(actions)

    def step_wait(self, **kwargs):
        next_obs, reward, term, trunc, info = self._envp.recv()

        if self.num_stack is not None:
            next_obs = self._unstack(next_obs)

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

    def __del__(self):
        self._envp.close()
