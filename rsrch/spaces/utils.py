import gymnasium as gym

from . import np as spaces_np


def from_gym(space: gym.Space) -> spaces_np.Array:
    """Convert a space from OpenAI's gym to space from here."""
    if isinstance(space, gym.spaces.Box):
        return spaces_np.Box(
            space.shape,
            low=space.low,
            high=space.high,
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return spaces_np.Discrete(
            space.n,
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Dict):
        return {k: from_gym(v) for k, v in space.items()}
    else:
        return spaces_np.Array(
            shape=space.shape,
            dtype=space.dtype,
        )


def to_gym(space: spaces_np.Array) -> gym.Space:
    """Convert a space from here to OpenAI gym's space format."""
    if isinstance(space, spaces_np.Box):
        return gym.spaces.Box(space.low, space.high, space.shape, space.dtype)
    elif isinstance(space, spaces_np.Discrete):
        return gym.spaces.Discrete(space.n)
    elif isinstance(space, dict):
        return gym.spaces.Dict({k: to_gym(v) for k, v in space.items()})
    else:
        return gym.spaces.Space(shape=space.shape, dtype=space.dtype)
