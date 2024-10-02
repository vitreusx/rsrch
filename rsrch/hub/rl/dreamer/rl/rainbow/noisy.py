import math
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsrch.nn.rewrite import rewrite_module_

_on_forward: Literal["zero", "reset", "none"] = "none"


@contextmanager
def on_forward(mode: Literal["zero", "reset", "none"]):
    """Change behavior of the noisy layer on forward call."""

    global _on_forward
    prev = _on_forward
    _on_forward = mode
    try:
        yield
    finally:
        _on_forward = prev


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma0: float,
        bias=True,
        factorized=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._sigma0 = sigma0
        self._bias = bias
        self._factorized = factorized

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.empty_like(self.weight))
        self.weight_eps: Tensor
        self.register_buffer("weight_eps", torch.empty_like(self.weight))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.noisy_bias = nn.Parameter(torch.empty(out_features))
            self.bias_eps: Tensor
            self.register_buffer("bias_eps", torch.empty_like(self.bias))

        self.init_weights()
        self.reset_noise_()

    def init_weights(self):
        s = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -s, s)
        nn.init.constant_(self.noisy_weight, self._sigma0 * s)
        if self._bias:
            nn.init.uniform_(self.bias, -s, s)
            nn.init.constant_(self.noisy_bias, self._sigma0 * s)

    @torch.no_grad()
    def reset_noise_(self):
        device, dtype = self.weight.device, self.weight.dtype

        if self._factorized:
            eps_in = torch.randn(self.in_features, device=device, dtype=dtype)
            sign_in = eps_in.sign()
            eps_in.abs_().sqrt_().mul_(sign_in)

            eps_out = torch.randn(self.out_features, device=device, dtype=dtype)
            sign_out = eps_out.sign()
            eps_out.abs_().sqrt_().mul_(sign_out)

            self.weight_eps.copy_(eps_out.outer(eps_in))
            if self._bias:
                self.bias_eps.copy_(eps_out)
        else:
            self.weight_eps.normal_()
            if self._bias:
                self.bias_eps.normal_()

    @torch.no_grad()
    def zero_noise_(self):
        self.weight_eps.zero_()
        if self._bias:
            self.bias_eps.zero_()

    def forward(self, x):
        if _on_forward == "zero":
            self.zero_noise_()
        elif _on_forward == "reset":
            self.reset_noise_()

        w = self.weight + self.noisy_weight * self.weight_eps
        if self._bias:
            b = self.bias + self.noisy_bias * self.bias_eps
            return F.linear(x, w, b)
        else:
            return F.linear(x, w)


def replace_all(
    module: nn.Module,
    sigma0: float,
    factorized: bool = True,
):
    """Replace all nn.Linear layers with NoisyLinear."""

    def _rewriter(module: nn.Module):
        if isinstance(module, nn.Linear):
            noisy = NoisyLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                sigma0=sigma0,
                factorized=factorized,
            )
            noisy.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                noisy.bias.data.copy_(module.bias.data)
            return noisy
        else:
            return module

    return rewrite_module_(module, _rewriter)
