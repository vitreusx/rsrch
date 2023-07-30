import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor) -> Tensor:
        # x.shape, E.shape = [..., D], [V, D]
        batch_dim, embed_dim = input.shape[:-1], input.shape[-1]
        input = input.reshape(-1, embed_dim)  # [N, D]
        # |x-e_i|^2 = <x-e_i,x-e_i>=<x,x>-2<x,e_i>+<e_i,e_i>
        # To minimize, we can skip <x, x>
        dists = -2 * F.linear(input, weight) + (weight * weight).sum(0).unsqueeze(
            0
        )  # [N, V]
        idxes = dists.argmin(1)  # [N,]
        values = weight[idxes].reshape(*batch_dim, embed_dim)
        return values

    @staticmethod
    def backward(ctx, values_grad: Tensor):
        input_grad = values_grad.clone()  # Straight-through gradients
        weight_grad = None
        return input_grad, weight_grad


def quantize(input: Tensor, weight: Tensor) -> Tensor:
    return Quantize.apply(input, weight)


class Quantizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weight: Tensor
        self.register_buffer("weight", torch.randn(vocab_size, embed_dim))

    def forward(self, input: Tensor) -> Tensor:
        return quantize(input, self.weight)
