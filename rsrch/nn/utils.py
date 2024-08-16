from contextlib import contextmanager
from functools import cache, wraps
from numbers import Number
from typing import TypeVar

import torch
from torch import Tensor, nn

from rsrch.types.tensorlike.core import Tensorlike


def safe_mode(*nets: nn.Module, enabled: bool = True):
    """Disables autograd and sets networks to eval model. Useful for computing shapes of outputs for modules.

    If no arguments are passed, functions as a decorator - otherwise, functions as a context manager.
    """

    @contextmanager
    def _safe_mode_ctx(*nets: nn.Module):
        if enabled:
            prev = [net.training for net in nets]
            for net in nets:
                net.eval()
            with torch.no_grad():
                yield
            for net, val in zip(nets, prev):
                net.train(mode=val)
        else:
            yield

    if len(nets) == 0:

        def decorator(_func):
            @wraps(_func)
            def _wrapped(self, *args, **kwargs):
                with _safe_mode_ctx(self):
                    return _func(self, *args, **kwargs)

            return _wrapped

        return decorator

    else:
        return _safe_mode_ctx(*nets)


def _flatten(x):
    if isinstance(x, tuple):
        x_f, shape = [], None
        for elem in x:
            elem_f, elem_shape = _flatten(elem)
            x_f.append(elem_f)
            if elem_shape is not None:
                shape = elem_shape
        return tuple(x_f), shape
    elif isinstance(x, dict):
        x_f, shape = {}, None
        for name in x:
            val_f, val_shape = _flatten(x[name])
            x_f[name] = val_f
            if val_shape is not None:
                shape = val_shape
        return x_f, shape
    elif hasattr(x, "flatten") and len(x.shape) >= 2:
        return x.flatten(0, 1), x.shape[:2]
    else:
        return x, None


def _unflatten(x, shape):
    if isinstance(x, tuple):
        return tuple(_unflatten(elem, shape) for elem in x)
    elif isinstance(x, dict):
        return {name: _unflatten(value, shape) for name, value in x.items()}
    elif hasattr(x, "reshape") and len(x.shape) >= 1:
        return x.reshape((*shape, *x.shape[1:]))
    else:
        return x


@cache
def _over_seq(_func):
    @wraps(_func)
    def _lifted(*args, **kwargs):
        args, shape = _flatten(args)
        res = _func(*args, **kwargs)
        res = _unflatten(res, shape)
        return res

    return _lifted


Func = TypeVar("Func")


def over_seq(_func: Func) -> Func:
    """Transform a function that operates on batches (N, ...) to operate on
    sequences (L, N, ...). The reshape takes place for positional arguments."""
    return _over_seq(_func)


class PassGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, to: Tensor):
        return value

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return None, grad_output


def pass_gradient(value: Tensor, to: Tensor) -> Tensor:
    if torch.jit.is_tracing():
        return value.detach() + (to - to.detach())
    else:
        return PassGradient.apply(value, to)
