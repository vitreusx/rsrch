from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import torch
from torch import Tensor

import rsrch.distributions as D
from rsrch.types.tensorlike import Tensorlike


class ValueDist(D.Distribution, Tensorlike):
    def __init__(
        self,
        v_min: Tensor | Number,
        v_max: Tensor | Number,
        N: int,
        index_rv: D.Categorical,
    ):
        Tensorlike.__init__(self, index_rv.shape)
        self.event_shape = torch.Size([])
        self.v_min = v_min
        self.v_max = v_max
        self.N = N

        self.index_rv: D.Categorical
        self.register("index_rv", index_rv)

    @property
    def device(self):
        return self.index_rv._param.device

    @property
    def dtype(self):
        return self.index_rv._param.dtype

    @property
    def probs(self):
        return self.index_rv.probs

    @property
    def logits(self):
        return self.index_rv.logits

    @property
    def log_probs(self):
        return self.index_rv.log_probs

    @property
    def mode(self):
        idx = self.index_rv._param.argmax(-1)
        t = idx / (self.N - 1)
        return self.v_min * (1.0 - t) + self.v_max * t

    @property
    def supp(self):
        if isinstance(self.v_min, Tensor):
            t = torch.linspace(0.0, 1.0, self.N, device=self.device, dtype=self.dtype)
            return torch.outer(self.v_min, 1.0 - t) + torch.outer(self.v_max, t)
        else:
            return torch.linspace(
                self.v_min, self.v_max, self.N, device=self.device, dtype=self.dtype
            )

    @property
    def mean(self):
        probs = self.index_rv.probs
        return (probs * self.supp).sum(-1)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        idx = self.index_rv.sample(sample_shape)
        t = idx / (self.N - 1)
        return self.v_min * (1.0 - t) + self.v_max * t

    def argmax(self, dim=None, keepdim=False):
        return self.mean.argmax(dim, keepdim)

    def gather(self, dim: int, index: Tensor):
        dim = range(len(self.shape))[dim]
        index = index[..., None]
        index = index.expand(*index.shape[:-1], self.N)
        if self.index_rv._probs is not None:
            probs = self.index_rv._probs.gather(dim, index)
            new_rv = D.Categorical(probs=probs)
        else:
            logits = self.index_rv._logits.gather(dim, index)
            new_rv = D.Categorical(logits=logits)
        return ValueDist(self.v_min, self.v_max, self.N, new_rv)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        onehot_rv = D.OneHotCategoricalST(
            probs=self.index_rv._probs,
            logits=self.index_rv._logits,
        )
        onehot = onehot_rv.rsample(sample_shape)
        grid = torch.linspace(
            self.v_min,
            self.v_max,
            self.N,
            dtype=onehot.dtype,
            device=onehot.device,
        )
        return (onehot * grid).sum(-1)

    def entropy(self):
        return self.index_rv.entropy()

    def __add__(self, dv):
        return self._with_supp(self.v_min + dv, self.v_max + dv)

    def _with_supp(self, new_v_min, new_v_max):
        return ValueDist(new_v_min, new_v_max, self.N, self.index_rv)

    def __radd__(self, dv):
        return self._with_supp(dv + self.v_min, dv + self.v_max)

    def __sub__(self, dv):
        return self._with_supp(self.v_min - dv, self.v_max - dv)

    def __mul__(self, scale):
        return self._with_supp(self.v_min * scale, self.v_max * scale)

    def __rmul__(self, scale: float):
        return self._with_supp(scale * self.v_min, scale * self.v_max)

    def __truediv__(self, div: float):
        return self._with_supp(self.v_min / div, self.v_max / div)

    @staticmethod
    def proj_kl_div(p: ValueDist, q: ValueDist):
        supp_p = p.supp.broadcast_to(*p.shape, p.N)  # [*B, N_p]
        supp_q = q.supp.broadcast_to(*q.shape, q.N)  # [*B, N_q]
        supp_p = supp_p.clamp(q.v_min[..., None], q.v_max[..., None])
        dz_q = (q.v_max - q.v_min) / (q.N - 1)
        dz_q = dz_q[..., None, None]
        supp_p, supp_q = supp_p[..., None, :], supp_q[..., None]
        t = (1 - (supp_p - supp_q).abs() / dz_q).clamp(0, 1)  # [*B, N_q, N_p]
        proj_probs = (t * p.probs[..., None, :]).sum(-1)  # [*B, N_q]
        kl_div = D.kl_divergence(D.Categorical(probs=proj_probs), q.index_rv)
        return kl_div
