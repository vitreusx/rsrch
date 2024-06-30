import rsrch.jax.nn.functional as F
from rsrch import jax
from rsrch.jax import Array, nn
from rsrch.jax import numpy as jnp
from rsrch.jax.random import keygen


class GRUCell(nn.Module):
    input_size: int = nn.final()
    hidden_size: float = nn.final()
    update_bias: float = nn.final()

    input_fc: nn.Linear
    hidden_fc: nn.Linear
    norm: nn.LayerNorm | None

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        update_bias: float = -1.0,
        norm: bool = False,
        *,
        key: Array
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.update_bias = update_bias

        key = keygen(key)
        self.input_fc = nn.Linear(input_size, 3 * hidden_size, key=next(key))
        self.hidden_fc = nn.Linear(hidden_size, 3 * hidden_size, key=next(key))
        if norm:
            self.norm = nn.LayerNorm(3 * hidden_size) if norm else None
        else:
            self.norm = None

    @jax.jit
    def __call__(self, input: Array, hidden: Array) -> Array:
        parts = self.input_fc(input) + self.hidden_fc(hidden)
        if self.norm:
            parts = self.norm(parts)
        reset, cand, update = jnp.split(parts, 3)
        cand = jnp.tanh(jax.nn.sigmoid(reset) * cand)
        update = F.sigmoid(update - self.update_bias) * hidden
        out = update * cand + (1.0 - update) * hidden
        return out


class State(jax.PyTree):
    deter: Array
    stoch: Array
