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


def pass_gradient(value: Tensor, to: Tensor) -> Tensor:
    return PassGradientOp.apply(value, to)


class ScaleGradientOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, scale: Number | Tensor):
        scale = torch.as_tensor(scale).type_as(value)
        ctx.save_for_backward(scale)
        return value

    @staticmethod
    def backward(ctx, grad_value: Tensor):
        (scale,) = ctx.saved_tensors
        return scale * grad_value


def scale_gradient(value: Tensor, scale: Number | Tensor):
    return ScaleGradientOp.apply(value, scale)
