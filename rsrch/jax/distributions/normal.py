import math

from rsrch import jax
from rsrch.jax import numpy as jnp

from .distribution import Distribution


class Normal(jax.PyTree, Distribution):
    loc: jax.Array
    scale: jax.Array

    def __init__(self, loc: jax.Array, scale: jax.Array):
        super().__init__()
        loc, scale = jnp.broadcast_arrays(loc, scale)
        self.loc = loc
        self.scale = scale

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def var(self):
        return jnp.square(self.scale)

    def rsample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()):
        eps = jax.random.normal(key, (*sample_shape, self.scale.shape))
        return self.loc + self.scale * eps

    def log_prob(self, value: jax.Array):
        output = (
            -(jnp.square(value - self.loc) / (2 * self.var))
            - jnp.log(self.scale)
            - 0.5 * math.log(2 * math.pi)
        )
        return output

    def entropy(self):
        ent = (0.5 + 0.5 * math.log(2 * math.pi)) + jnp.log(self.scale)
        return ent

    @staticmethod
    def kl_div(lhs: "Normal", rhs: "Normal"):
        ratio = lhs.var / rhs.var
        value = (
            jnp.square(lhs.loc - rhs.loc) / (2 * rhs.var)
            + (ratio - 1.0 - jnp.log(ratio)) / 2
        )
        return value
