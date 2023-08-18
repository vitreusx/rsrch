import torch
import torch.nn as nn
from torch import Tensor


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.main = nn.Linear(in_features, out_features, bias=bias)
        self.noisy = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        weight_noise = torch.randn_like(self.noisy.weight)
        self.noisy.weight *= weight_noise

        if self.noisy.bias is not None:
            bias_noise = torch.randn_like(self.noisy.bias)
            self.noisy.bias *= bias_noise

        return self.main(x) + self.noisy(x)
