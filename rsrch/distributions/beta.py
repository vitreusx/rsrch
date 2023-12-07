import torch
from torch import Tensor

from rsrch.types import Tensorlike

from .distribution import Distribution
from .utils import sum_rightmost


class Beta(Distribution, Tensorlike):
    def __init__(self, alpha: Tensor, beta: Tensor, event_dims: int = 0):
        shape = alpha.shape
        pivot = len(shape) - event_dims
        batch_shape, event_shape = shape[:pivot], shape[pivot:]
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = event_shape

        self.alpha = self.register("alpha", alpha)
        self.beta = self.register("beta", beta)
        self._log_B = self.register("_log_B", None)

    @property
    def log_B(self):
        if self._log_B is None:
            t1 = torch.special.gammaln(self.alpha)
            t2 = torch.special.gammaln(self.beta)
            t3 = torch.special.gammaln(self.alpha + self.beta)
            self._log_B = t1 + t2 - t3
        return self._log_B

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def var(self):
        t1 = self.alpha * self.beta
        t2 = (self.alpha + self.beta).square()
        t3 = self.alpha + self.beta + 1.0
        return t1 / (t2 * t3)

    def log_prob(self, value: Tensor):
        t1 = (self.alpha - 1.0) * value.log()
        t2 = (self.beta - 1.0) * (1.0 - value).log()
        logp = t1 + t2 - self.log_B
        return sum_rightmost(logp, len(self.event_shape))

    def entropy(self):
        t1 = (self.alpha - 1.0) * torch.special.digamma(self.alpha)
        t2 = (self.beta - 1.0) * torch.special.digamma(self.beta)
        t3 = (self.alpha + self.beta - 2.0) * torch.special.digamma(
            self.alpha + self.beta
        )
        ent = self.log_B - t1 - t2 + t3
        return sum_rightmost(ent, len(self.event_shape))
