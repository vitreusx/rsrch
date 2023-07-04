from typing import List, Literal

import torch.nn as nn


class FullyConnected(nn.Sequential):
    def __init__(
        self,
        num_features: List[int],
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.ReLU,
        final_layer: Literal["fc", "norm", "act"] = "fc",
    ):
        layers = []
        in_out = enumerate(zip(num_features[:-1], num_features[1:]))
        final_idx = len(num_features) - 2

        for idx, (in_features, out_features) in in_out:
            if norm_layer is None:
                fc = nn.Linear(in_features, out_features, bias=True)
                act = act_layer()
                layers.extend((fc, act))
            else:
                fc_bias = (idx == final_idx) and (final_layer == "fc")
                fc = nn.Linear(in_features, out_features, bias=fc_bias)
                norm = norm_layer(out_features)
                act = act_layer()
                layers.extend((fc, norm, act))

        if norm_layer is None:
            peel_off = {"fc": 1, "act": 0}
        else:
            peel_off = {"fc": 2, "norm": 1, "act": 0}
        peel_off = peel_off[final_layer]
        if peel_off > 0:
            layers = layers[:-peel_off]

        super().__init__(*layers)
