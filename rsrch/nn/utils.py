import torch
from torch import Tensor


class ST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, *grad_targets: Tensor):
        ctx.save_for_backward(*grad_targets)
        return value

    @staticmethod
    def backward(ctx, out_grad: Tensor):
        grad_targets = ctx.saved_tensors
        return tuple([None, *(out_grad for _ in grad_targets)])
