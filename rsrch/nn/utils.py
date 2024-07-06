from contextlib import contextmanager
from functools import cache, wraps
from numbers import Number
from typing import TypeVar

import torch
from torch import Tensor, nn


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


@cache
def _over_seq(_func):
    @wraps(_func)
    def _lifted(*args, **kwargs):
        seq_len, batch_size = args[0].shape[:2]
        y = _func(*(x.flatten(0, 1) for x in args), **kwargs)
        if isinstance(y, tuple):
            xs = []
            for x in y:
                if len(x.shape) > 0:
                    x = x.reshape(seq_len, batch_size, *x.shape[1:])
                xs.append(x)
            y = tuple(xs)
        else:
            if len(y.shape) > 0:
                y = y.reshape(seq_len, batch_size, *y.shape[1:])
        return y

    return _lifted


Func = TypeVar("Func")


def over_seq(_func: Func) -> Func:
    """Transform a function that operates on batches (N, ...) to operate on
    sequences (L, N, ...). The reshape takes place for positional arguments."""
    return _over_seq(_func)


def pass_gradient(value: Tensor, to: Tensor) -> Tensor:
    return value.detach() + (to - to.detach())
