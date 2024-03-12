from contextlib import contextmanager

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
def infer_ctx(*nets: nn.Module):
    """Disable autograd and set networks' mode to 'eval'."""
    prev = [net.training for net in nets]
    for net in nets:
        net.eval()
    with torch.inference_mode():
        yield
    for net, val in zip(nets, prev):
        net.train(mode=val)
