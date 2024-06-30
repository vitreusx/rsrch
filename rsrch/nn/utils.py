from contextlib import contextmanager
from functools import wraps

import torch
from torch import Tensor, nn


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, copy_grad_to: Tensor):
        return value

    @staticmethod
    def backward(ctx, value_grad: Tensor):
        return None, value_grad


def straight_through(value: Tensor, copy_grad_to: Tensor):
    """Pass a value unchanged in the forward pass, but redirect backprop to another tensor."""
    return StraightThrough.apply(value, copy_grad_to)


@contextmanager
def _inference_eval_ctx(*nets: nn.Module):
    prev = [net.training for net in nets]
    for net in nets:
        net.eval()
    with torch.no_grad():
        yield
    for net, val in zip(nets, prev):
        net.train(mode=val)


def _inference_eval_decorator(_func):
    @wraps(_func)
    def _wrapped(self, *args, **kwargs):
        with _inference_eval_ctx(self):
            return _func(self, *args, **kwargs)

    return _wrapped


def safe_mode(*nets: nn.Module):
    """Disables autograd and sets networks to eval model. Useful for computing shapes of outputs for modules.

    If no arguments are passed, functions as a decorator - otherwise, functions as a context manager.
    """

    if len(nets) == 0:
        return _inference_eval_decorator
    else:
        return _inference_eval_ctx(*nets)


from functools import cache, wraps
from numbers import Number
from typing import TypeVar

import torch
from torch import Tensor


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


class PassGradientOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, to: Tensor):
        return value

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return None, grad_output


# def pass_gradient(value: Tensor, to: Tensor) -> Tensor:
#     return PassGradientOp.apply(value, to)


def pass_gradient(value: Tensor, to: Tensor) -> Tensor:
    return value.detach() + (to - to.detach())
