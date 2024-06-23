from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import *
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.distributions.utils import sum_rightmost
from rsrch.exp import Experiment
from rsrch.types import Tensorlike
from rsrch.utils.preview import make_grid

from . import data
from .tokenizer import InputSpec, Tokenizer
from .utils import over_seq, pass_gradient


class WorldModel(nn.Module):
    def __init__(self, input_spec: InputSpec):
        super().__init__()
        self.spec = input_spec

        self.obs_features = 128
        self._obs_emb = nn.Embedding(
            self.spec.obs.vocab_size,
            self.obs_features,
        )
        self.obs_size = self.obs_features * self.spec.obs.num_tokens

        self.act_features = 64
        self._act_emb = nn.Embedding(
            self.spec.act.vocab_size,
            self.act_features,
        )
        self.act_size = self.act_features * self.spec.act.num_tokens

        self.top_lm = nn.GRU(
            self.obs_size + self.act_size,
            1024,
        )
