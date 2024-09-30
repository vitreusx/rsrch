from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import safe_mode

from ..common import nets
from ..common.trainer import TrainerBase
from ..common.types import Slices


@dataclass
class Config:
    @dataclass
    class Actor:
        opt: dict
        opt_ratio: int
        hidden_dim: int
        num_layers: int
        norm_layer: nets.NormType
        act_layer: nets.ActType

    @dataclass
    class Q:
        opt: dict
        hidden_dim: int
        num_layers: int
        norm_layer: nets.NormType
        act_layer: nets.ActType

    actor: Actor
    qf: Q
    num_q: int
    gamma: float


class Q(nn.Module):
    def __init__(
        self,
        cfg: Config.Q,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = int(np.prod(act_space.shape))

        self.fc = nets.MLP(
            in_features=obs_dim + act_dim,
            out_features=1,
            hidden=cfg.hidden_dim,
            layers=cfg.num_layers,
            norm=cfg.norm_layer,
            act=cfg.act_layer,
        )

    def forward(self, obs: Tensor, act: Tensor):
        x = torch.cat([obs.flatten(1), act.flatten(1)], -1)
        return self.fc(x).flatten(1)


class Actor(nn.Sequential):
    def __init__(
        self,
        cfg: Config.Actor,
        obs_space: spaces.torch.Tensor,
        act_space: spaces.torch.Tensor,
    ):
        obs_dim = int(np.prod(obs_space.shape))
        fc = nets.MLP(
            in_features=obs_dim,
            out_features=None,
            hidden=cfg.hidden_dim,
            layers=cfg.num_layers,
            norm=cfg.norm_layer,
            act=cfg.act_layer,
        )

        with safe_mode(fc):
            input = obs_space.sample((1,)).flatten(1)
            out_features = fc(input).shape[1]

        head = dh.make(
            layer_ctor=partial(nn.Linear, out_features),
            space=act_space,
        )

        super().__init__(nn.Flatten(1), fc, head)


class Trainer(TrainerBase):
    def __init__(
        self,
        cfg: Config,
        actor: Actor,
        make_q: Callable[[], Q],
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(compute_dtype)
        self.cfg = cfg
        self.actor = actor

        self.qf, self.qf_t = nn.ModuleList(), nn.ModuleList()
        for _ in range(cfg.num_q):
            self.qf.append(make_q())
            self.qf_t.append(make_q())

        self.actor_opt = self._make_opt(self.actor.parameters(), cfg.actor.opt)
        self.qf_opt = self._make_opt(self.qf.parameters(), cfg.qf.opt)

        self.opt_iter = 0

    def _make_opt(self, parameters: list[nn.Parameter], cfg: dict):
        cls = getattr(torch.optim, cfg["type"])
        del cfg["type"]
        return cls(parameters, **cfg)

    def compute(self, batch: Slices):
        with torch.no_grad():
            with self.autocast():
                next_obs = batch.obs[-1]
                next_policy = self.actor(next_obs)
                next_act = next_policy.sample()
                min_q = self.qf_t[0](next_obs, next_act)
                for idx in range(1, self.cfg.num_q):
                    min_q = torch.min(min_q, self.qf_t[idx](next_obs, next_act))
