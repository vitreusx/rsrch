from jax import Array, random
from jax.random import *


def keygen(seed: Array):
    """Create an infinite sequence of PRNG keys."""

    while True:
        seed, key = random.split(seed)
        yield key
