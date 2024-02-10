import torch
from torch import Tensor


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
