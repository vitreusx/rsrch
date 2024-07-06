import jax


class Distribution:
    @property
    def mean(self) -> jax.Array:
        raise NotImplementedError()

    @property
    def variance(self) -> jax.Array:
        raise NotImplementedError()

    def log_prob(self, value: jax.Array) -> jax.Array:
        raise NotImplementedError()

    def entropy(self) -> jax.Array:
        raise NotImplementedError()

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()):
        value = self.rsample(key, sample_shape)
        return jax.lax.stop_gradient(value)

    def rsample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()):
        raise NotImplementedError()
