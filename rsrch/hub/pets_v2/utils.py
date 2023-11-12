from functools import cache

from torch import Tensor


@cache
def over_seq(_func):
    """Transform a function that operates on batches (B, ...) to operate on
    sequences (L, B, ...)."""

    def _lifted(x: Tensor, *args, **kwargs):
        batch_size = x.shape[1]
        y = _func(x.flatten(0, 1), *args, **kwargs)
        return y.reshape(-1, batch_size, *y.shape[1:])

    return _lifted
