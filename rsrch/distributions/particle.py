import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.types import Tensorlike


class Particle(D.Distribution, Tensorlike):
    """A finite-support distribution over reals."""

    def __init__(self, atoms: Tensor, indices: D.Categorical):
        Tensorlike.__init__(self, indices.shape)
        self.atoms = self.register("atoms", atoms)
        self.indices = self.register("indices", indices)

    @property
    def mean(self) -> Tensor:
        p = self.indices.probs
        return (self.atoms * p).sum(-1)  # [*B]

    @property
    def mode(self) -> Tensor:
        max_idx = self.indices.mode  # [*B]
        return self.atoms.gather(-1, max_idx.unsqueeze(-1)).squeeze(-1)

    @property
    def var(self) -> Tensor:
        dx = self.atoms - self.mean.unsqueeze(-1)
        var_i = dx.square().sum(-2)  # [*B, K]
        p = self.indices.probs
        return (var_i * p).sum(-1)  # [*B]

    def entropy(self) -> Tensor:
        # This "fake entropy" formula is meant to replicate the entropy of
        # continuous uniform distribution U[a, b] as K -> \inf and the atoms
        # are distributed "uniformly", i.e. evenly spaced and with equal prob.
        return 0.5 * (self.var.log() + np.log(12))

    def log_prob(self, value: Tensor) -> Tensor:
        # For a given value, we find the closest atom and return associated
        # log prob.
        dists = (value.unsqueeze(-1) - self.atoms).square()
        idxes = dists.argmin(-1)  # [*S, *B]
        return self.indices.log_prob(idxes)
