import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["Linear"]


class Linear(nn.Module):
    """An ensemble of nn.Linear layers - allows one to do the entire
    computation in a single pass."""

    class Slice(nn.Module):
        """A proxy for a single model within the ensemble."""

        def __init__(self, weight: Tensor, bias: Tensor | None):
            super().__init__()
            self.weight = weight
            self.bias = bias

        def forward(self, input: Tensor):
            return F.linear(input, self.weight, self.bias)

    def __init__(self, in_features: int, out_features: int, num_models=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_models = num_models

        self.weight = nn.Parameter(torch.empty((num_models, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_models, 1, out_features)))
        else:
            self.register_parameter("bias", None)

        self.slices = nn.ModuleList()
        for idx in range(self.num_models):
            weight = self.weight[idx].T
            bias = self.bias[idx, 0] if self.bias is not None else None
            self.slices.append(Linear.Slice(weight, bias))

        self.reset_parameters()

    def __getitem__(self, idx):
        return self.slices[idx]

    def reset_parameters(self):
        for idx in range(self.num_models):
            # This section is virtually the same as in nn.Linear
            nn.init.kaiming_uniform_(self.weight[idx].T, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[idx].T)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx][0], -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # input.shape = [num_models, batch_size, in_features]
        # output.shape = [num_models, batch_size, out_features]
        output = torch.bmm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output
