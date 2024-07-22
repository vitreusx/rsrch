from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from rsrch import spaces

from . import nets
from .utils import find_class, null_ctx
from .wm import WorldModel


@dataclass
class Config:
    opt: dict
    target_critic: dict | None
    rew_norm: dict


class Actor(nn.Module):
    def __init__(
        self,
        state_size: int,
        act_space: spaces.torch.Space,
        cfg: Config,
    ):
        super().__init__()
        self.mlp = nets.MLP(state_size, None, **cfg.actor)
        self.head = nets.ActorHead(self.mlp.out_features, act_space)

    def forward(self, state):
        ...


class Trainer(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        make_critic: Callable[[], nn.Module],
        cfg: Config,
        compute_dtype: torch.dtype | None,
    ):
        super().__init__()
        self.actor = actor
        self.cfg = cfg
        self.compute_dtype = compute_dtype

        self.critic = make_critic()

        if cfg.target_critic is not None:
            self.target_critic = make_critic()
            polyak.sync(self.critic, self.target_critic)
            freeze_(self.target_critic)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.target_critic,
                **cfg.target_critic,
            )
        else:
            self.target_critic = self.critic

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)

        self.opt = self._make_opt()
        self._device = next(self.actor.parameters()).device
        self.scaler = getattr(torch, self._device.type).amp.GradScaler()

    def _make_opt(self):
        cfg = {**self.cfg.opt}

        cls = find_class(torch.optim, cfg["$type"])
        del cfg["$type"]

        actor, critic = cfg["actor"], cfg["critic"]
        del cfg["actor"]
        del cfg["critic"]

        common = cfg

        return cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.critic.parameters(), **critic},
            ],
            **common,
        )

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self._device.type,
                dtype=self.compute_dtype,
            )

    def _loss_fn(self, batch: dict):
        mets = {}
