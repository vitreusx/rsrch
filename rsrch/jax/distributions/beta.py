from rsrch import jax
import jax.numpy as jnp

from .distribution import Distribution


class Beta(jax.PyTree, Distribution):
    alpha: jax.Array
    beta: jax.Array

    def __init__(self, alpha: jax.Array, beta: jax.Array):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def mode(self):
        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    @property
    def var(self):
        t1 = self.alpha * self.beta
        t2 = jnp.square(self.alpha + self.beta)
        t3 = self.alpha + self.beta + 1.0
        return t1 / (t2 * t3)

    def rsample(self, key: jax.Array, sample_shape=()):
        shape = (*sample_shape, *self.alpha.shape)
        return jax.random.beta(key, self.alpha, self.beta, shape)

    def log_prob(self, value: jax.Array):
        t1 = (self.alpha - 1.0) * jnp.log(value)
        t2 = (self.beta - 1.0) * jnp.log(1.0 - value)
        return t1 + t2 - jax.scipy.special.betaln(self.alpha, self.beta)

    def entropy(self):
        digamma = jax.scipy.special.digamma
        t1 = (self.alpha - 1.0) * digamma(self.alpha)
        t2 = (self.beta - 1.0) * digamma(self.beta)
        t3 = (self.alpha + self.beta - 2.0) * digamma(self.alpha + self.beta)
        logB = jax.scipy.special.betaln(self.alpha, self.beta)
        return logB - t1 - t2 + t3
