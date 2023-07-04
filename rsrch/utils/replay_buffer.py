import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    act: torch.Tensor
    next_obs: torch.Tensor
    reward: torch.Tensor
    trunc: torch.Tensor
    term: torch.Tensor

    def to(self, device: torch.device):
        return TransitionBatch(
            obs=self.obs.to(device),
            act=self.act.to(device),
            next_obs=self.next_obs.to(device),
            reward=self.reward.to(device),
            trunc=self.trunc.to(device),
            term=self.term.to(device),
        )

    def __len__(self):
        return len(self.obs)


class ReplayBuffer(nn.Module):
    def __init__(self, obs, act, capacity):
        super().__init__()

        self.obs = torch.empty((capacity, *obs.shape), dtype=obs.dtype)
        self.act = torch.empty((capacity, *act.shape), dtype=act.dtype)
        self.next_obs = torch.empty_like(self.next_obs)
        self.reward = torch.empty((capacity,), dtype=torch.float32)
        self.trunc = torch.empty((capacity,), dtype=torch.bool)
        self.term = torch.empty((capacity,), dtype=torch.bool)
        self.size, self._cur_idx, self.capacity = 0, 0, capacity

    def push(self, batch: TransitionBatch):
        n = len(batch)
        idxes = torch.arange(self._cur_idx, self._cur_idx + n)
        idxes = idxes % self.capacity

        self.obs[idxes] = batch.obs
        self.act[idxes] = batch.act
        self.next_obs[idxes] = batch.next_obs
        self.reward[idxes] = batch.reward
        self.trunc[idxes] = batch.trunc
        self.term[idxes] = batch.term

        self.size = min(self.size + n, self.capacity)
        self._cur_idx = (self._cur_idx + n) % self.capacity

    def sample(self, n):
        idxes = torch.randint(self.size, n)

        return TransitionBatch(
            obs=self.obs[idxes],
            act=self.act[idxes],
            next_obs=self.next_obs[idxes],
            reward=self.reward[idxes],
            trunc=self.trunc[idxes],
            term=self.term[idxes],
        )
