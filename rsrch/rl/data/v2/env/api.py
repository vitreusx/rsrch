from rsrch.rl import gym


class Factory:
    def env(self, **kwargs) -> gym.Env:
        ...

    def vec_env(self, num_envs: int, **kwargs) -> gym.VectorEnv:
        ...
