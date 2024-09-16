from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import rsrch.distributions as D
from rsrch.nn.utils import pass_gradient
from rsrch.types import Tensorlike


def invperm(idxes: Tensor) -> Tensor:
    irange = torch.arange(idxes.shape[-1], dtype=idxes.dtype, device=idxes.device)
    irange = irange.expand_as(idxes)
    invperm = torch.empty_like(idxes)
    invperm.scatter_(-1, idxes, irange)
    return invperm


class ValueDist(Tensorlike, D.Distribution):
    def __init__(
        self,
        ind_rv: D.Categorical,
        v_min: Number | Tensor,
        v_max: Number | Tensor,
    ):
        Tensorlike.__init__(self, ind_rv.shape)
        self.ind_rv = self.register("ind_rv", ind_rv)
        self.N = ind_rv.num_events
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
        indices = pass_gradient(indices.float(), grad_target)

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

    @staticmethod
    def proj_kl_div(p: ValueDist, q: ValueDist):
        q_vmin, q_vmax = q.v_min.unsqueeze(-1), q.v_max.unsqueeze(-1)
        dz = (q_vmax - q_vmin) / q.N  # [*B, 1]
        idx = ((p.support - q_vmin) / dz).clamp(0, q.N - 1)  # [*B, p.N]
        idx_f, idx_c = idx.floor(), idx.ceil()
        m = torch.zeros_like(q.ind_rv.probs)  # [*B, q.N]
        frac_c = idx - idx_f
        m.scatter_add_(-1, idx_f.long(), p.ind_rv.probs * (1.0 - frac_c))
        m.scatter_add_(-1, idx_c.long(), p.ind_rv.probs * frac_c)
        return D.kl_divergence(D.Categorical(probs=m), q.ind_rv)

    @staticmethod
    def w1_div(p: ValueDist, q: ValueDist):
        sup = torch.cat((p.support, q.support), dim=-1)
        sup, idxes = torch.sort(sup, dim=-1)
        idxes = invperm(idxes)

        p_pr = p.ind_rv.probs
        p_cdf = torch.zeros_like(sup, requires_grad=p_pr.requires_grad)
        p_cdf = p_cdf.scatter(-1, idxes[..., : p.N], p_pr)
        p_cdf = p_cdf.cumsum(-1)

        q_pr = q.ind_rv.probs
        q_cdf = torch.zeros_like(sup, requires_grad=q_pr.requires_grad)
        q_cdf = q_cdf.scatter(-1, idxes[..., p.N :], q_pr)
        q_cdf = q_cdf.cumsum(-1)

        delta = torch.diff(sup, dim=-1)
        return ((p_cdf - q_cdf).abs()[..., :-1] * delta).sum(-1)
