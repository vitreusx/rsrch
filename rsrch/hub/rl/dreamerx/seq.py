import torch
from torch import nn, Tensor


class RecurWM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
        num_layers: int,
        deter_dim: int,
        stoch_dim: int,
    ):
        super().__init__()

        self.init_fc = nn.Linear(obs_dim, hidden_size * num_layers)
        self.infer_rnn = nn.GRU(obs_dim + act_dim, hidden_size, num_layers)
