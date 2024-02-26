import envpool
import numpy as np

from rsrch.rl import gym


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
        self._envp.send(actions)

    def step_wait(self, **kwargs):
        next_obs, reward, term, trunc, info = self._envp.recv()
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
