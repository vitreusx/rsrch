import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from dataclasses import dataclass
from typing import Tuple, Union, overload


class LSTMCell(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = self.H = hidden_features
        self.bias = bias

        self.input_map = nn.Linear(in_features, 4 * hidden_features, bias=bias)
        self.state_map = nn.Linear(hidden_features, 4 * hidden_features, bias=bias)

    def forward(self, x: Tensor, h_c: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # x.shape = [B, N_in]
        h, c = h_c
        gates = self.state_map(h) + self.input_map(x)  # [B, 4*H]
        i, f, g, o = torch.split(gates, self.H, dim=1)  # [B, H] each
        next_c = F.sigmoid(f) * c + F.sigmoid(i) * F.tanh(g)  # [B, H]
        next_h = F.sigmoid(o) * next_c  # [B, H]
        return next_h, next_c


class LSTM(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias=True):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.bias = bias

        self.cell = LSTMCell(in_features, hidden_features, bias)

    def _forward_tensor(
        self, xs: Tensor, h_c: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # x.shape = [L, B, N_in]
        h, c = h_c  # [B, H]
        hs = []
        for x in xs:
            h, c = self.cell(x, (h, c))
            hs.append(h)
        hs = torch.stack(hs)
        return hs, (h, c)

    def _forward_packed(
        self, seq: rnn.PackedSequence, h_c: Tuple[Tensor, Tensor]
    ) -> Tuple[rnn.PackedSequence, Tuple[Tensor, Tensor]]:
        h, c = h_c  # [N, H]
        h, c = h[seq.sorted_indices], c[seq.sorted_indices]
        hs = torch.empty(len(seq.data), self.hidden_features)
        cur_offset = 0
        for step_batch in seq.batch_sizes:
            cur_slice = slice(cur_offset, cur_offset + step_batch)
            h[:step_batch], c[:step_batch] = self.cell(
                seq.data[cur_slice],
                (h[:step_batch], c[:step_batch]),
            )
            hs[cur_slice] = h[:step_batch]
            cur_offset += step_batch
        h_seq = rnn.PackedSequence(
            hs, seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices
        )
        h, c = h[seq.unsorted_indices], c[seq.unsorted_indices]
        return h_seq, (h, c)

    @overload
    def forward(
        self, seq: Tensor, h_c: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        ...

    @overload
    def forward(
        self, seq: rnn.PackedSequence, h_c: Tuple[Tensor, Tensor]
    ) -> Tuple[rnn.PackedSequence, Tuple[Tensor, Tensor]]:
        ...

    def forward(self, seq, h_c):
        if isinstance(seq, Tensor):
            return self._forward_tensor(seq, h_c)
        elif isinstance(seq, rnn.PackedSequence):
            return self._forward_packed(seq, h_c)
