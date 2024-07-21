import gymnasium as gym

from . import np


def from_gym(space: gym.Space) -> np.Space:
    """Convert a space from OpenAI's gym to space from here."""
    if isinstance(space, gym.spaces.Box):
        return np.Box(
            space.shape,
            low=space.low,
            high=space.high,
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return np.Discrete(
            space.n,
            dtype=space.dtype,
        )
    else:
        raise NotImplementedError(type(space))


def to_gym(space: np.Space) -> gym.Space:
    """Convert a space from here to OpenAI gym's space format."""
    if isinstance(space, np.Box):
        return gym.spaces.Box(space.low, space.high, space.shape, space.dtype)
    elif isinstance(space, np.Discrete):
        return gym.spaces.Discrete(space.n)
    else:
        raise NotImplementedError(type(space))
