import numpy as np

from rsrch.rl import gym

from .base import Factory


class RandomSeedWrapper:
    """A wrapper around a factory, which ensures that the spaces and the envs
    are properly seeded. Otherwise, one can encounter determinism issues."""

    def __init__(self, factory: Factory, seed: int):
        self.factory = factory
        self.seed = seed
        self._seed_seq = np.random.SeedSequence(self.seed)

    def _next_seed(self) -> int:
        return self._seed_seq.spawn(1)[0].generate_state(1)[0]

    def env(self, **kwargs):
        env = self.factory.env(**kwargs)
        env.reset(seed=self._next_seed())
        env.observation_space.seed(self._next_seed())
        env.action_space.seed(self._next_seed())
        return env

    def vec_env(self, num_envs: int, **kwargs):
        vec_env = self.factory.vec_env(num_envs, **kwargs)
        vec_env.observation_space.seed(self._next_seed())
        vec_env.single_observation_space.seed(self._next_seed())
        vec_env.action_space.seed(self._next_seed())
        vec_env.single_action_space.seed(self._next_seed())
        vec_env.reset(seed=self._next_seed())
        return vec_env


def is_stacked(env: gym.Env | gym.VectorEnv):
    """Check whether the env returns stacked observations (see
    gym.wrappers.FrameStack for more details.) Can be used to optimize storing
    the environment steps in a buffer."""

    return getattr(env, "stack_num", None) is not None
