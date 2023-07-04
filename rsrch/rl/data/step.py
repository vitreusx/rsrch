from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch
from torch import Tensor


@dataclass
class Step:
    obs: Any
    act: Any
    next_obs: Any
    reward: float
    term: bool


@dataclass
class TensorStep:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: float
    term: bool

    def to(self, device: torch.device) -> TensorStep:
        return TensorStep(
            obs=self.obs.to(device),
            act=self.act.to(device),
            next_obs=self.next_obs.to(device),
            reward=self.reward,
            term=self.term,
        )


def to_tensor_step(step: Step) -> TensorStep:
    return TensorStep(
        obs=torch.as_tensor(step.obs),
        act=torch.as_tensor(step.act),
        next_obs=torch.as_tensor(step.next_obs),
        reward=step.reward,
        term=step.term,
    )


@dataclass
class StepBatch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device: torch.device):
        return StepBatch(
            self.obs.to(device),
            self.act.to(device),
            self.next_obs.to(device),
            self.reward.to(device),
            term=self.term.to(device),
        )

    @staticmethod
    def collate(batch: List[TensorStep]):
        obs = torch.stack([x.obs for x in batch])
        act = torch.stack([x.act for x in batch])
        next_obs = torch.stack([x.next_obs for x in batch])

        device = obs.device
        reward = torch.tensor([x.reward for x in batch], device=device)
        term = torch.tensor([x.term for x in batch], device=device)

        return StepBatch(obs, act, next_obs, reward, term)
