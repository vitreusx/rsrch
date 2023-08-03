from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Generic, SupportsFloat, TypeVar

import torch
from torch import Tensor

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class Step(Generic[ObsType, ActType]):
    obs: ObsType
    act: ActType
    next_obs: ObsType
    reward: SupportsFloat
    term: SupportsFloat


@dataclass
class TensorStep(Step[Tensor, Tensor]):
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: SupportsFloat
    term: SupportsFloat

    @staticmethod
    def convert(step: Step) -> TensorStep:
        return TensorStep(
            obs=torch.as_tensor(step.obs).detach(),
            act=torch.as_tensor(step.act).detach(),
            next_obs=torch.as_tensor(step.next_obs).detach(),
            reward=step.reward,
            term=step.term,
        )

    def to(self, device: torch.device) -> TensorStep:
        return TensorStep(
            obs=self.obs.to(device),
            act=self.act.to(device),
            next_obs=self.next_obs.to(device),
            reward=self.reward,
            term=self.term,
        )


@dataclass
class StepBatch:
    obs: Tensor
    act: Tensor
    next_obs: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device: torch.device) -> StepBatch:
        return StepBatch(
            self.obs.to(device),
            self.act.to(device),
            self.next_obs.to(device),
            self.reward.to(device),
            term=self.term.to(device),
        )

    @staticmethod
    def collate_fn(batch: typing.Sequence[TensorStep]) -> StepBatch:
        if not isinstance(batch, list):
            batch = [*batch]

        obs = torch.stack([x.obs for x in batch])
        act = torch.stack([x.act for x in batch])
        next_obs = torch.stack([x.next_obs for x in batch])

        device = obs.device
        reward = torch.tensor([x.reward for x in batch], device=device)
        term = torch.tensor([x.term for x in batch], device=device)

        return StepBatch(obs, act, next_obs, reward, term)
