import numpy as np
import torch
import torch.nn as nn

from rsrch.datasets.transforms import Compose, Transform
from rsrch.rl.data.seq import Sequence, TensorSeq

from ..step import Step, TensorStep
from . import functional as F


class Subsample(nn.Module, Transform):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, seq: Sequence):
        start = np.random.randint(len(seq))
        end = start + self.max_seq_len + 1
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
