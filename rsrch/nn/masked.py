import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias: nn.Parameter | None = None
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.register_buffer("ext_mult", 1.0)
        self.register_buffer("core_features", self.in_features)

    def expand_(self, in_features: int, out_features: int):
        self.core_features = self.in_features
        self.in_features = in_features
        self.out_features = out_features
        self.ext_mult = 0.0

        prev_weight = self.weight.data
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight[:, : self.core_features] = prev_weight

        return self

    def forward(self, x: Tensor) -> Tensor:
        cf = self.core_features
        core_out = F.linear(x[:cf], self.weight[:, :cf], self.bias)
        ext_out = F.linear(x[cf:], self.weight[:, cf:], self.bias)
        return core_out + self.ext_mult * ext_out
