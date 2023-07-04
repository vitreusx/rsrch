import torch
import torch.nn as nn
import numpy as np
from rsrch.rl.data.trajectory import Trajectory, TensorTrajectory


class Compose(nn.Module):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class Subsample(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, seq: Trajectory):
        start = np.random.randint(len(seq))
        end = start + self.max_seq_len
        return seq[start:end]


class ToTensorSeq(nn.Module):
    def forward(self, seq: Trajectory):
        return TensorTrajectory.from_seq(seq)


class PadTensorSeq(nn.Module):
    def __init__(self, min_seq_len: int):
        super().__init__()
        self.min_seq_len = min_seq_len

    def forward(self, seq: TensorTrajectory):
        cur_len = len(seq)
        if cur_len >= self.min_seq_len:
            return seq

        obs = torch.stack([seq.obs[-1]] * self.min_seq_len)
        obs[:cur_len] = seq.obs
        act = torch.stack([seq.act[-1]] * self.min_seq_len)
        act[:cur_len] = seq.act
        reward = torch.stack([seq.reward[-1]] * self.min_seq_len)
        reward[:cur_len] = seq.reward

        return TensorTrajectory(obs, act, reward, seq.term, seq.trunc)
