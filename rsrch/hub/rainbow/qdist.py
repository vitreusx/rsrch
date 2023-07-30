import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits

import rsrch.distributions as D
import rsrch.distributions.constraints as C


class ParticleDist(D.Distribution):
    arg_constraints = {
        "probs": C.simplex,
        "logits": C.real,
        "particles": C.real,
    }

    def __init__(
        self,
        particles: Tensor,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args=False,
    ):
        self._probs, self._logits = probs, logits
        if probs is not None:
            self.probs = probs
            batch_shape = probs.shape[:-1]
        else:
            self.logits = logits
            batch_shape = logits.shape[:-1]

        event_shape = particles.shape[len(batch_shape) + 1 :]
        super().__init__(batch_shape, event_shape, validate_args)

        self._batch_size = self.B = int(np.prod(self.batch_shape))
        self._num_classes = self.C = particles.shape[len(batch_shape)]
        self._event_size = self.N = int(np.prod(self.event_shape))

        self.particles = particles
        self._flat = particles.reshape(self.B, self.C, self.N)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs)

    @lazy_property
    def log_probs(self) -> Tensor:
        return self.logits - self.logits.logsumexp(dim=-1, keepdim=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits)

    @lazy_property
    def onehot_dist(self):
        return D.OneHotCategoricalStraightThrough(
            probs=self.probs,
            validate_args=False,
        )

    @lazy_property
    def idx_dist(self):
        return D.Categorical(
            probs=self.probs,
            validate_args=False,
        )

    def log_prob(self, samples: Tensor):
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()):
        S = int(np.prod(sample_shape))
        idx: Tensor = self.idx_dist.sample(sample_shape)
        # idx.shape = [*sample_shape, *batch_shape]
        idx = idx.reshape(S, self.B, 1, 1).expand(S, self.B, 1, self.N)
        # idx.shape = [S, B, 1, N]
        flat = self._flat.unsqueeze(0).expand(S, self.B, self.C, self.N)
        # flat.shape = [S, B, C, N]
        sample = flat.gather(2, idx).squeeze(2)
        # sample.shape = [S, B, N]
        return sample.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        S = int(np.prod(sample_shape))
        onehot = self.onehot_dist.rsample(sample_shape)
        # onehot.shape = [*sample_shape, *batch_shape, C]
        onehot = onehot.reshape(S, self.B, self.C)
        # onehot.shape = [S, B, C]
        # self._flat.shape = [B, C, N]
        sample = torch.einsum("sbc,bcn->sbn", onehot, self._flat)
        # sample.shape = [S, B, N]
        return sample.reshape(*sample_shape, *self.batch_shape, *self.event_shape)

    def _new(self, new_particles):
        return ParticleDist(
            new_particles,
            probs=self._probs,
            logits=self._logits,
            validate_args=False,
        )

    def __add__(self, other):
        return self._new(self.particles + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._new(self.particles - other)

    def __rsub__(self, other):
        return self._new(other - self.particles)

    def __mul__(self, other):
        return self._new(self.particles * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._new(self.particles / other)

    def __rtruediv__(self, other):
        return self._new(other / self.particles)


def orig_kl_div(p: ParticleDist, q: ParticleDist):
    if len(p.event_shape) > 0 or len(q.event_shape) > 0:
        raise NotImplementedError

    # assert issorted(p.particles) and issorted(q.particles)
    v_min, v_max = q.particles[0], q.particles[-1]
    delta_z = (v_max - v_min) / (len(q.particles) - 1)
    p_ = p.particles.clamp(v_min, v_max)
    q_ = q.particles.reshape(-1, 1)
    proj = ((1 - (p_ - q_).abs() / delta_z).clamp(0, 1) * p.probs).sum(0)
    proj = D.Categorical(probs=proj)
    return D.kl_divergence(proj, q.idx_dist)


def quasi_div(p: ParticleDist, q: ParticleDist, alpha=0.5):
    if len(p.event_shape) > 0 or len(q.event_shape) > 0:
        raise NotImplementedError

    sort_p, idx_p = torch.sort(p.particles)
    sort_q, idx_q = torch.sort(q.particles)
    emb_loss = (sort_p - sort_q).abs().sum(-1)

    pmf_p = p.probs.gather(-1, idx_p)
    cdf_p = torch.cumsum(pmf_p, -1)
    pmf_q = q.probs.gather(-1, idx_q)
    cdf_q = torch.cumsum(pmf_q, -1)
    w1_loss = (cdf_p - cdf_q).abs().sum(-1)

    return alpha * emb_loss + (1.0 - alpha) * w1_loss
