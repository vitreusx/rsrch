import numpy as np
import torch
import torch.nn as nn

from rsrch.datasets.transforms import Compose, Transform

from ..step import Step, TensorStep
from ..trajectory import TensorTrajectory, Trajectory
from . import functional as F


class Subsample(nn.Module, Transform):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, seq: Trajectory):
        start = np.random.randint(len(seq))
        end = start + self.max_seq_len
        return seq[start:end]


class ToTensorSeq(nn.Module, Transform):
    def __init__(self):
        super().__init__()

    def forward(self, seq: Trajectory):
        return F.to_tensor_seq(seq)


class ToTensorStep(nn.Module, Transform):
    def __init__(self):
        super().__init__()

    def forward(self, step: Step) -> TensorStep:
        return F.to_tensor_step(step)


class PadTensorSeq(nn.Module, Transform):
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

        return TensorTrajectory(obs, act, reward, seq.term)
