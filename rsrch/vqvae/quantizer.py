import torch
from torch import Tensor
import torch.nn as nn


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, E: Tensor) -> Tensor:
        # x.shape, E.shape = [..., D], [V, D]
        ctx.save_for_backward(E)

        assert x.shape[-1] == E.shape[-1]
        layout, vector_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, vector_dim)  # [N, D]
        dists = -2 * (x @ E.T) + (E**2).sum(0).unsqueeze(0)  # [N, V]
        idxes = dists.argmin(1)  # [N,]
        values = E[idxes].reshape(*layout, vector_dim)
        return values

    @staticmethod
    def backward(ctx, values_grad: Tensor):
        (E,) = ctx.saved_tensors
        x_grad = values_grad.clone()
        E_grad = torch.zeros_like(E)
        return x_grad, E_grad


class Quantizer(nn.Module):
    def __init__(self, vocab_size: int, vector_dim: int):
        super().__init__()
        self._vocab_size = vocab_size
        self._vector_dim = vector_dim

        self.E: Tensor
        self.register_buffer("E", torch.randn(vocab_size, vector_dim))

    def forward(self, x: Tensor) -> Tensor:
        return Quantize.apply(x)  # type: ignore
