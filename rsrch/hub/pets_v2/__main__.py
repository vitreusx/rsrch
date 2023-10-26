import torch
from torch import nn, Tensor
from rsrch.nn import fc
from . import config, env
from pathlib import Path


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, num_layers: int):
        super().__init__()

        self.init = fc.FullyConnected(
            layer_sizes=[obs_dim, hidden, hidden],
            norm_layer=None,
            act_layer=nn.ReLU,
        )

        self.seq = nn.GRU(
            input_size=act_dim,
            hidden_size=hidden,
            num_layers=num_layers,
        )

        self.rv: nn.Module

    def forward(self, obs: Tensor, act: Tensor):
        # obs.shape = [N, D_obs]
        # act.shape = [L, N, D_act]
        L, N = act.shape[:2]
        init_h = self.init(obs)  # [N, H]
        init_h = init_h.expand(self.seq.num_layers, *init_h.shape)  # [L, N, H]
        out, _ = self.seq(act, init_h)  # [L, N, H]
        rvs = self.rv(out.flatten(0, 1))  # [L*N, *]
        rvs = rvs.reshape(L, N, *rvs.shape[1:])  # [L, N, *]
        return rvs


def main():
    cfg_dict = config.from_args(
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    cfg = config.to_class(
        data=cfg_dict,
        cls=config.Config,
    )

    env_f = env.make_factory(cfg.env)

    ...

    while True:
        ...


if __name__ == "__main__":
    main()
