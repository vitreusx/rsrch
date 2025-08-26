from typing import Iterable

import jax


def key_seq(key: jax.Array) -> Iterable[jax.Array]:
    while True:
        key, sub = jax.random.split(key)
        yield sub
