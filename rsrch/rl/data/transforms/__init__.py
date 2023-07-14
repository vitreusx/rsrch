import numpy as np
import torch
import torch.nn as nn

from rsrch.datasets.transforms import Compose, Transform
from rsrch.rl.data.seq import Sequence, TensorSeq

from ..step import Step, TensorStep
from . import functional as F


class Subsample(nn.Module, Transform):
    def __init__(
        self,
        min_seq_len: int,
        max_seq_len: int | None = None,
        prioritize_ends: bool = False,
    ):
        super().__init__()
        self.min_seq_len = min_seq_len
        if max_seq_len is None:
            max_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.prioritize_ends = prioritize_ends

    def forward(self, seq: Sequence):
        assert len(seq) >= self.min_seq_len
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len + 1)
        max_obs_idx, slice_len = len(seq) + 1, seq_len + 1
        if self.prioritize_ends:
            start = np.random.randint(max_obs_idx)
            start = min(start, max_obs_idx - slice_len)
        else:
            start = np.random.randint(max_obs_idx - slice_len)
        end = start + slice_len
        return seq[start:end]


class ToTensorSeq(nn.Module, Transform):
    def __init__(self):
        super().__init__()

    def forward(self, seq: Sequence) -> TensorSeq:
        return F.to_tensor_seq(seq)


class ToTensorStep(nn.Module, Transform):
    def __init__(self):
        super().__init__()

    def forward(self, step: Step) -> TensorStep:
        return F.to_tensor_step(step)


class ToDevice(nn.Module, Transform):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(device=self.device)
