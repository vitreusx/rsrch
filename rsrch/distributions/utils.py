from torch import Tensor


def sum_rightmost(value: Tensor, k: int):
    if k > 0:
        value = value.flatten(len(value.shape) - k).sum(-1)
    return value
