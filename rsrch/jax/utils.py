from typing import Iterable

import jax


def key_seq(key: jax.Array | None) -> Iterable[jax.Array | None]:
    """Get a sequence of keys for initializing JAX modules."""
    while True:
        if key is None:
            yield None
        else:
            key, sub = jax.random.split(key)
            yield sub
