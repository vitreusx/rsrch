import torch
import torch.nn as nn
from torch import Tensor


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, E: Tensor) -> Tensor:
        # x.shape, E.shape = [..., D], [V, D]
        assert x.shape[-1] == E.shape[-1]
        batch_dim, embed_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, embed_dim)  # [N, D]
        # |x-e_i|^2 = <x-e_i,x-e_i>=<x,x>-2<x,e_i>+<e_i,e_i>
        # To minimize, we can skip <x, x>
        dists = -2 * (x @ E.T) + (E**2).sum(0).unsqueeze(0)  # [N, V]
        idxes = dists.argmin(1)  # [N,]
        values = E[idxes].reshape(*batch_dim, embed_dim)
        return values

    @staticmethod
    def backward(ctx, values_grad: Tensor):
        x_grad = values_grad.clone()  # Straight-through
        return x_grad, None


class Quantizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.E: Tensor
        self.register_buffer("E", torch.randn(vocab_size, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        return Quantize.apply(x)  # type: ignore
