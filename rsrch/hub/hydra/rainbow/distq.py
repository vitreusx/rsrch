from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import rsrch.distributions as D
from rsrch.nn.utils import straight_through
from rsrch.types import Tensorlike


class ValueDist(Tensorlike, D.Distribution):
    def __init__(
        self,
        ind_rv: D.Categorical,
        v_min: Number | Tensor,
        v_max: Number | Tensor,
    ):
        Tensorlike.__init__(self, ind_rv.shape)
        self.ind_rv = self.register("ind_rv", ind_rv)
        self.N = ind_rv._num_events
        v_min = torch.as_tensor(v_min, device=self.device).expand(self.shape)
        self.v_min = self.register("v_min", v_min)
        v_max = torch.as_tensor(v_max, device=self.device).expand(self.shape)
        self.v_max = self.register("v_max", v_max)

    @property
    def mode(self):
        t = (self.ind_rv.mode / (self.N - 1))[..., None]
        return self.v_min * (1.0 - t) + self.v_max * t

    @property
    def support(self):
        t = torch.linspace(0.0, 1.0, self.N, device=self.device)
        return self.v_min[..., None] * (1.0 - t) + self.v_max[..., None] * t

    @property
    def mean(self):
        return (self.ind_rv.probs * self.support).sum(-1)

    def sample(self, sample_shape: tuple[int, ...] = ()):
        indices = self.ind_rv.sample(sample_shape)
        t = indices / (self.N - 1)
        return (self.v_min * (1.0 - t) + self.v_max * t).detach()

    def rsample(self, sample_shape: tuple[int, ...] = ()):
        num_samples = int(np.prod(sample_shape))
        indices = self.ind_rv.sample([num_samples])  # [#S, *B]

        grad_target = self.ind_rv._param  # [*B, N]
        grad_target = grad_target[None].expand(num_samples, *self.batch_shape, self.N)
        grad_target = grad_target.gather(-1, indices[..., None]).squeeze(-1)
        indices = straight_through(indices.float(), grad_target)

        t = indices / (self.N - 1)
        values = self.v_min[None] * (1.0 - t) + self.v_max[None] * t
        values = values.reshape(*sample_shape, *self.batch_shape)

        return values

    def __add__(self, other):
        return ValueDist(self.ind_rv, self.v_min + other, self.v_max + other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        assert isinstance(other, Number) and other > 0
        return ValueDist(self.ind_rv, self.v_min * other, self.v_max * other)

    def __rmul__(self, other):
        return self * other

    def argmax(self, dim=None, keepdim=False):
        return self.mean.argmax(dim=dim, keepdim=keepdim)

    @staticmethod
    def proj_kl_div(p: ValueDist, q: ValueDist):
        supp_p = p.support.broadcast_to(*p.shape, p.N)  # [*B, N_p]
        supp_q = q.support.broadcast_to(*q.shape, q.N)  # [*B, N_q]
        supp_p = supp_p.clamp(q.v_min[..., None], q.v_max[..., None])
        dz_q = (q.v_max - q.v_min) / (q.N - 1)
        dz_q = dz_q[..., None, None]
        supp_p, supp_q = supp_p[..., None, :], supp_q[..., None]
        t = (1 - (supp_p - supp_q).abs() / dz_q).clamp(0, 1)  # [*B, N_q, N_p]
        proj_probs = (t * p.ind_rv.probs[..., None, :]).sum(-1)  # [*B, N_q]
        proj_ind_rv = D.Categorical(probs=proj_probs)
        return D.kl_divergence(proj_ind_rv, q.ind_rv)


@dataclass
class Config:
    enabled: bool
    num_atoms: int
    v_min: float
    v_max: float
