import torch
from torch import Tensor, nn

import rsrch.distributions as D


class DistLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: tuple[int, ...],
        dist_type: str,
        disc_heads: int,
        std_act: str,
        init_std: float,
        min_std: float,
    ):
        super().__init__()
        self.discrete = discrete
        self.stoch = stoch
        self.std_act = std_act
        self.min_std = min_std
        if isinstance(discrete, int):
            self._fc = nn.Linear(in_features, stoch * discrete)
        else:
            self._fc = nn.Linear(in_features, 2 * stoch)

    def forward(self, x: Tensor):
        out = self._fc(x)
        if self.discrete:
            return D.MultiheadOHST(self.discrete, logits=out)
        else:
            mean, std = out.split(2, -1)
            if self.std_act == "softplus":
                std = nn.functional.softplus(std)
            elif self.std_act == "sigmoid":
                std = nn.functional.sigmoid(std)
            elif self.std_act == "sigmoid2":
                std = 2.0 * nn.functional.sigmoid(0.5 * std)
            else:
                raise ValueError(self.std_act)
            return D.Normal(mean, std + self.min_std, 1)
