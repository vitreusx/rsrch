from typing import List, Literal

import torch.nn as nn


class FullyConnected(nn.Sequential):
    def __init__(
        self,
        layer_sizes: List[int],
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.ReLU,
        final_layer: Literal["fc", "norm", "act"] = "fc",
    ):
        layers = []
        in_out = enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        final_idx = len(layer_sizes) - 2

        layer_types = []
        for idx, (in_features, out_features) in in_out:
            norm = None
            if norm_layer is not None:
                norm = norm_layer(out_features)

            if norm is None or isinstance(norm, nn.Identity):
                fc = nn.Linear(in_features, out_features, bias=True)
                act = act_layer()
                layers.extend((fc, act))
                layer_types.extend(("fc", "act"))
            else:
                fc_bias = (idx == final_idx) and (final_layer == "fc")
                fc = nn.Linear(in_features, out_features, bias=fc_bias)
                act = act_layer()
                layers.extend((fc, norm, act))
                layer_types.extend(("fc", "norm", "act"))

        while layer_types[-1] != final_layer:
            layers.pop()
            layer_types.pop()

        super().__init__(*layers)
