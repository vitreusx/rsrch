from rsrch.rl import gym

from . import np


def from_gym(space: gym.Space) -> np.Space:
    """Convert a space from OpenAI's gym to space from here."""
    if isinstance(space, gym.spaces.Box):
        return np.Box(space.shape, space.low, space.high, space.dtype, space._np_random)
    elif isinstance(space, gym.spaces.Discrete):
        return np.Discrete(space.n, space.dtype, space._np_random)
    else:
        raise NotImplementedError(type(space))


def to_gym(space: np.Space) -> gym.Space:
    """Convert a space from here to OpenAI gym's space format."""
    if isinstance(space, np.Box):
        return gym.spaces.Box(
            space.low, space.high, space.shape, space.dtype, space._seed
        )
    elif isinstance(space, np.Discrete):
        return gym.spaces.Discrete(space.n, space._seed)
    else:
        raise NotImplementedError(type(space))
