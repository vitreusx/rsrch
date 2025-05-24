from typing import Callable, List, Literal

import torch
import torch.nn as nn
from torch import Tensor


class SkipLinear(nn.Module):
    def __init__(self, features: int, bias=True):
        super().__init__()
        self.H = nn.Linear(features, features, bias=bias)
        self.T = nn.Linear(features, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        t = torch.sigmoid(self.T(x))
        return t * self.H(x) + (1.0 - t) * x


class FullyConnected(nn.Sequential):
    def __init__(
        self,
        layer_sizes: List[int],
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        final_layer: Literal["fc", "norm", "act"] = "fc",
        highway: bool = False,
    ):
        layers = []
        in_out = enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        final_idx = len(layer_sizes) - 2

        layer_types = []
        for idx, (in_features, out_features) in in_out:
            fc_bias = norm_layer is None or (
                (idx == final_idx) and (final_layer == "fc")
            )
            if highway and in_features == out_features:
                fc = SkipLinear(in_features, bias=fc_bias)
            else:
                fc = nn.Linear(in_features, out_features, bias=fc_bias)

            act = act_layer()

            if norm_layer is None:
                layers.extend((fc, act))
                layer_types.extend(("fc", "act"))
            else:
                norm = norm_layer(out_features)
                act = act_layer()
                layers.extend((fc, norm, act))
                layer_types.extend(("fc", "norm", "act"))

        while layer_types[-1] != final_layer:
            layers.pop()
            layer_types.pop()

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        super().__init__(*layers)
