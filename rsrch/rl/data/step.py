from dataclasses import dataclass
from typing import Any, List, Protocol
import torch
from torch import Tensor


@dataclass
class Step:
    obs: Any
    act: Any
    next_obs: Any
    reward: float
    term: bool
    trunc: bool


class TensorStep(Protocol):
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: float
    term: bool
    trunc: bool


@dataclass
class StepBatch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor
    trunc: Tensor

    def to(self, device: torch.device):
        return StepBatch(
            self.obs.to(device),
            self.act.to(device),
            self.next_obs.to(device),
            self.reward.to(device),
            term=self.term.to(device),
            trunc=self.trunc.to(device),
        )

    @staticmethod
    def collate(batch: List[TensorStep]):
        obs = torch.stack([x.obs for x in batch])
        act = torch.stack([x.act for x in batch])
        next_obs = torch.stack([x.next_obs for x in batch])

        device = obs.device
        reward = torch.tensor([x.reward for x in batch], device=device)
        term = torch.tensor([x.term for x in batch], device=device)
        trunc = torch.tensor([x.trunc for x in batch], device=device)

        return StepBatch(obs, act, next_obs, reward, term, trunc)
