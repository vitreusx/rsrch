import torch
from torch import nn, Tensor

import rsrch.distributions as D
from rsrch.types import Tensorlike


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def zero_(self):
        self.deter.zero_()
        self.stoch.zero_()
        return self

    def to_tensor(self):
        return torch.cat((self.deter, self.stoch), -1)


class StateDist(D.Distribution, Tensorlike):
    def __init__(
        self,
        *,
        deter: Tensor,
        stoch: D.Distribution,
    ):
        Tensorlike.__init__(self, stoch.batch_shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def sample(self, sample_shape=()):
        deter = self.deter.expand(*sample_shape, *self.deter.shape)
        stoch = self.stoch.sample(sample_shape)
        return State(deter, stoch)

    def rsample(self, sample_shape=()):
        deter = self.deter.expand(*sample_shape, *self.deter.shape)
        stoch = self.stoch.rsample(sample_shape)
        return State(deter, stoch)

    @property
    def mode(self):
        return State(self.deter, self.stoch.mode)

    def entropy(self):
        return self.stoch.entropy()


@D.register_kl(StateDist, StateDist)
def _(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch, q.stoch)


class EnsembleRSSM(nn.Module)