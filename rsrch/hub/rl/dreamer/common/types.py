from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Slices:
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device: torch.device | None = None):
        return Slices(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            term=self.term.to(device),
        )

    def pin_memory(self):
        return Slices(
            obs=self.obs.pin_memory(),
            act=self.act.pin_memory(),
            reward=self.reward.pin_memory(),
            term=self.term.pin_memory(),
        )
