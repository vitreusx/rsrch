import rsrch.jax.nn.functional as F
from rsrch import jax
from rsrch.jax import numpy as jnp

from .distribution import Distribution


class Categorical(jax.PyTree, Distribution):
    probs: jax.Array
    logits: jax.Array
    log_probs: jax.Array
    batch_shape: tuple[int, ...] = jax.final()

    def __init__(
        self,
        *,
        probs: jax.Array | None = None,
        logits: jax.Array | None = None,
        log_probs: jax.Array | None = None,
    ):
        super().__init__()

        if sum(x is not None for x in (probs, logits, log_probs)) != 1:
            msg = "Precisely one of (probs, logits, log_probs) must be non-None."
            raise ValueError(msg)

        if probs is not None:
            self.probs = probs
            self.log_probs = self.logits = jnp.log(probs)
            self.batch_shape = probs.shape[:-1]
        elif logits is not None:
            self.probs = F.softmax(logits, axis=-1)
            self.logits = logits
            self.log_probs = logits - F.logsumexp(logits, -1, keepdims=True)
            self.batch_shape = logits.shape[:-1]
        elif log_probs is not None:
            self.probs = jnp.exp(log_probs)
            self.logits = self.log_probs = log_probs
            self.batch_shape = log_probs.shape[:-1]

    def sample(self, key: jax.Array, sample_shape=()):
        output_shape = (*sample_shape, *self.batch_shape)
        return jax.random.categorical(key, self.logits, output_shape)

    def log_prob(self, value: jax.Array):
        if value.dtype.kind == "f":
            return jnp.sum(value * self.log_probs, -1)
        else:
            value = jnp.expand_dims(value, -1)
            value, log_pmf = jnp.broadcast_arrays(value, self.log_probs)
            value = value[..., :1]
            output = jnp.take_along_axis(log_pmf, value, -1)
            return output[..., 0]

    def entropy(self):
        value = self.probs * self.log_probs
        return jnp.sum(value, -1)

    @staticmethod
    def kl_div(lhs: "Categorical", rhs: "Categorical"):
        value = lhs.probs * (lhs.log_probs - rhs.log_probs)
        return jnp.sum(value, -1)
