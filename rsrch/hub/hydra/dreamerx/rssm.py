from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.types import Tensorlike


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        Tensorlike.__init__(self, stoch.shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)


class StateDist(Tensorlike):
    def __init__(self, deter: Tensor, stoch: D.Distribution):
        Tensorlike.__init__(self, stoch.batch_shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)


class DistLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        stoch: int,
        discrete: int | bool,
        std_act="softplus",
        min_std=0.1,
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


class EnsembleRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=True,
        act="elu",
        norm=None,
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()

        act_layer = {"elu": nn.ELU, "relu": nn.ReLU}[act]
        norm_layer = {None: lambda _: nn.Identity, "bn": nn.BatchNorm1d}[norm]
        dist_layer = partial(
            DistLayer,
            stoch=stoch,
            discrete=discrete,
            std_act=std_act,
            min_std=min_std,
        )

        self._img_in = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden),
            norm_layer(hidden),
            act_layer(),
        )

        self._img_stoch = nn.ModuleList()
        for _ in range(ensemble):
            self._img_stoch.append(
                nn.Sequential(
                    nn.Linear(deter, hidden),
                    norm_layer(hidden),
                    act_layer(),
                    dist_layer(hidden),
                )
            )

        self._cell = nn.GRUCell(hidden, deter)

    def img_step(self, prev_h: State, act: Tensor):
        x = torch.cat([prev_h.stoch, act], -1)
        x = self._img_in(x)
        deter = self._cell(x, prev_h.deter)
        if self.training:
            index = torch.randint(0, len(self._img_stoch), size=()).item()
        else:
            index = 0
        stoch = self._img_stoch[index](deter)
        return StateDist(deter, stoch)
