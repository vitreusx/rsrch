from jax.random import *

from rsrch.jax import Array


def keygen(seed: Array):
    """Create an infinite sequence of PRNG keys."""

    while True:
        seed, key = split(seed)
        yield key
