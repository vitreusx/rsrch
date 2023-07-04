import torch.nn as nn
from typing import List, Literal


class FCN(nn.Sequential):
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
            fc_bias = (idx == final_idx) and (final_layer == "fc")
            layers.extend(
                [
                    nn.Linear(in_features, out_features, bias=fc_bias),
                    norm_layer(out_features),
                    act_layer(),
                ]
            )

        peel_off = {"fc": 2, "norm": 1, "act": 0}[final_layer]
        layers = layers[:-peel_off]

        super().__init__(*layers)
