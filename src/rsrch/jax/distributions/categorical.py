import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class Categorical(eqx.Module):
    probs: jax.Array | None = None
    logits: jax.Array | None = None
    log_probs: jax.Array | None = None

    def __init__(
        self,
        *,
        probs: jax.Array | None = None,
        logits: jax.Array | None = None,
        log_probs: jax.Array | None = None,
    ):
        super().__init__()
        self.probs = probs
        self.logits = logits
        self.log_probs = log_probs

    @jax.jit
    def get_probs(self):
        if self.probs is not None:
            return self.probs
        else:
            if self.logits is not None:
                probs = jax.nn.softmax(self.logits, axis=-1)
            else:
                probs = jnp.exp(self.log_probs)
            return probs

    def with_probs(self, probs: jax.Array):
        return Categorical(
            probs=probs,
            logits=self.logits,
            log_probs=self.log_probs,
        )

    @jax.jit
    def get_logits(self):
        if self.logits is not None:
            return self.logits
        else:
            if self.log_probs is not None:
                logits = self.log_probs
            else:
                logits = jnp.log(self.probs)
            return logits

    def with_logits(self, logits: jax.Array):
        return Categorical(
            probs=self.probs,
            logits=logits,
            log_probs=self.log_probs,
        )

    @jax.jit
    def get_log_probs(self):
        if self.log_probs is not None:
            return self.log_probs, self
        else:
            if self.logits is not None:
                log_probs = self.logits - jax.nn.logsumexp(
                    self.logits, axis=-1, keepdims=True
                )
            else:
                log_probs = jnp.log(self.probs)
            return log_probs

    def with_log_probs(self, log_probs: jax.Array):
        return Categorical(
            probs=self.probs,
            logits=self.logits,
            log_probs=log_probs,
        )

    @jax.jit
    def mean(self):
        raise NotImplementedError

    @jax.jit
    def mode(self):
        if self.log_probs is not None:
            return self.log_probs.argmax(axis=-1)
        elif self.logits is not None:
            return self.logits.argmax(axis=-1)
        elif self.probs is not None:
            return self.probs.argmax(axis=-1)
        raise ValueError

    def sample(self, *, key: jax.Array):
        logits = self.get_logits()
        eps = jnp.finfo(logits.dtype).eps
        unif = jax.random.uniform(key, shape=logits.shape, minval=eps, maxval=1.0 - eps)
        samples = (logits - jnp.log(-jnp.log(unif))).argmax(-1)
        return samples

    @jax.jit
    @jax.vmap
    def log_prob(self, value: jax.Array):
        logits = self.get_logits()
        num_classes = logits.shape[-1]
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(1, num_classes),
            value[None],
        )
        return -ce.sum()

    @jax.jit
    def entropy(self):
        log_p = self.get_log_probs()
        eps = jnp.finfo(log_p.dtype).eps
        log_neg_p_log_p = log_p + jnp.log((-log_p).clip(min=eps))
        ent = jnp.exp(jax.nn.logsumexp(log_neg_p_log_p, axis=-1))
        return ent.sum()
