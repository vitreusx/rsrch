from functools import partial

from rsrch import jax
from rsrch.jax import numpy as jnp

from .distribution import Distribution


@partial(jax.jit, static_argnames=["num_axes"])
def sum_rightmost(value: jax.Array, num_axes: int):
    if num_axes > 0:
        value = value.reshape(*value.shape[:-num_axes], -1)
        value = jnp.sum(value, -1)
    return value


class Independent(jax.PyTree, Distribution):
    base: Distribution
    num_axes: int = jax.final()

    def __init__(self, base: Distribution, num_axes: int):
        super().__init__()
        self.base = base
        self.num_axes = num_axes

    @property
    def mean(self):
        return self.base.mean

    @property
    def variance(self):
        return self.base.variance

    def sample(self, key: jax.Array, sample_shape=()):
        return self.base.sample(key, sample_shape)

    def rsample(self, key: jax.Array, sample_shape=()):
        return self.base.rsample(key, sample_shape)

    def log_prob(self, value: jax.Array):
        return sum_rightmost(self.base.log_prob(value), self.num_axes)

    def entropy(self):
        return sum_rightmost(self.base.entropy(), self.num_axes)

    @staticmethod
    def kl_div(lhs: "Independent", rhs: "Independent"):
        assert lhs.num_axes == rhs.num_axes
        return sum_rightmost(lhs.kl_div(lhs, rhs), lhs.num_axes)
