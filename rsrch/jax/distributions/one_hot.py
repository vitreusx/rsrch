import rsrch.jax.nn.functional as F
from rsrch import jax
from rsrch.jax import numpy as jnp

from .categorical import Categorical
from .distribution import Distribution


class OneHot(jax.PyTree, Distribution):
    index_dist: Categorical

    def __init__(self, index_dist: Categorical):
        super().__init__()
        self.index_dist = index_dist

    @property
    def mean(self):
        return self.index_dist.probs

    @property
    def mode(self):
        probs = self.index_dist.probs
        return F.one_hot(
            probs.argmax(-1),
            probs.shape[-1],
            dtype=probs.dtype,
        )

    def sample(self, key: jax.Array, sample_shape=()):
        indices = self.index_dist.sample(key, sample_shape)
        probs = self.index_dist.probs
        value = F.one_hot(indices, probs.shape[-1], dtype=probs.dtype)
        return value

    def rsample(self, key: jax.Array, sample_shape=()):
        indices = self.index_dist.sample(key, sample_shape)
        probs = self.index_dist.probs
        value = F.one_hot(indices, probs.shape[-1], dtype=probs.dtype)
        value = value + (probs - jax.lax.stop_gradient(probs))
        return value

    def log_prob(self, value: jax.Array):
        indices = value.argmax(-1)
        value = self.index_dist.log_prob(indices)
        return value

    @staticmethod
    def kl_div(lhs: "OneHot", rhs: "OneHot"):
        return Categorical.kl_div(lhs.index_dist, rhs.index_dist)
