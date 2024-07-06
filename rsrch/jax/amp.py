from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import jax.numpy as jnp
import jmp
from jax import lax, nn
from optax import losses

policy = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)


class autocast:
    def __init__(
        self,
        compute_dtype: jnp.dtype,
        output_dtype: jnp.dtype | None = None,
    ):
        self.compute_dtype = compute_dtype
        self.output_dtype = compute_dtype or output_dtype

    def __enter__(self):
        global policy
        self._prev_policy = policy
        policy = jmp.Policy(
            param_dtype=policy.param_dtype,
            compute_dtype=self.compute_dtype,
            output_dtype=self.output_dtype,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global policy
        policy = self._prev_policy


P, R = ParamSpec("P"), TypeVar("R")


def autocast_to_fp32(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapped(*args, **kwargs):
        args, kwargs = policy.cast_to_param((args, kwargs))
        res = func(*args, **kwargs)
        res = policy.cast_to_output(res)
        return res

    return wrapped


def autocast_to_fp16(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapped(*args, **kwargs):
        args, kwargs = policy.cast_to_compute((args, kwargs))
        res = func(*args, **kwargs)
        res = policy.cast_to_output(res)
        return res

    return wrapped
