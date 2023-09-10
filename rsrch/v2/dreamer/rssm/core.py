import torch
from torch import nn, Tensor
import rsrch.distributions as D
from rsrch.types import Tensorlike
import torch
import rsrch.distributions as D
from torch import Tensor
from .. import wm


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def as_tensor(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], -1)


class StateDist(Tensorlike, D.Distribution):
    def __init__(self, deter: Tensor, stoch_rv: D.Distribution):
        Tensorlike.__init__(self, shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch_rv = self.register("stoch_rv", stoch_rv)

    @property
    def mean(self):
        return State(self.deter, self.stoch_rv.mean)

    @property
    def mode(self):
        return State(self.deter, self.stoch_rv.mode)

    def sample(self, sample_shape) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.sample(sample_shape)
        return State(deter, stoch)

    def rsample(self, sample_shape) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.rsample(sample_shape)
        return State(deter, stoch)


@D.register_kl(StateDist, StateDist)
def _(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch_rv, q.stoch_rv)


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def as_tensor(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], -1)


class StateDist(Tensorlike, D.Distribution):
    def __init__(self, deter: Tensor, stoch_rv: D.Distribution):
        Tensorlike.__init__(self, shape=deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch_rv = self.register("stoch_rv", stoch_rv)

    @property
    def mean(self):
        return State(self.deter, self.stoch_rv.mean)

    @property
    def mode(self):
        return State(self.deter, self.stoch_rv.mode)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.sample(sample_shape)
        return State(deter, stoch)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> State:
        deter = self.deter.expand([*sample_shape, *self.deter.shape])
        stoch = self.stoch_rv.rsample(sample_shape)
        return State(deter, stoch)


@D.register_kl(StateDist, StateDist)
def _(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch_rv, q.stoch_rv)


class RSSM(wm.WorldModel):
    obs_enc: nn.Module
    act_enc: nn.Module
    deter_in: nn.Module
    deter_cell: nn.Module
    prior_stoch: nn.Module
    post_stoch: nn.Module

    def act_cell(self, prev_h: State, act):
        deter_x = torch.cat([prev_h.stoch, act], 1)
        deter_x = self.deter_in(deter_x)
        deter = self.deter_cell(deter_x, prev_h.deter)
        stoch_rv = self.prior_stoch(deter)
        state_rv = StateDist(deter, stoch_rv)
        return state_rv

    def obs_cell(self, prior: State, obs) -> StateDist:
        stoch_x = torch.cat([prior.deter, obs], 1)
        stoch_rv = self.post_stoch(stoch_x)
        return StateDist(prior.deter, stoch_rv)
