from functools import wraps
from typing import Final, List

from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.nn.utils import over_seq
from rsrch.types import Tensorlike

from . import config, nets
from .nets import *


class GRUCell(nn.Module):
    __constants__ = ["input_size", "hidden_size", "update_bias", "norm"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        update_bias: float = -1.0,
        norm: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.update_bias = update_bias
        self.norm = norm

        self._fc = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=not norm)
        self._norm = nn.LayerNorm(3 * hidden_size)

    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        parts: Tensor = self._fc(torch.cat((input, hidden), -1))
        if self.norm:
            parts = self._norm(parts).to(parts.dtype)
        reset, cand, update = parts.chunk(3, -1)
        cand = torch.tanh(reset.sigmoid() * cand)
        update = (update + self.update_bias).sigmoid()
        out = update * cand + (1 - update) * hidden
        return out


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


class EnsembleRSSM(nn.Module):
    __constants__ = [
        "obs_size",
        "act_size",
        "deter_size",
        "num_tokens",
        "token_size",
        "stoch_size",
    ]

    def __init__(self, cfg: config.RSSM, obs_size: int, act_size: int):
        super().__init__()
        self.cfg = cfg
        self.obs_size = obs_size
        self.act_size = act_size

        if not (
            cfg.ensemble == 1
            and cfg.norm == "none"
            and cfg.act == "elu"
            and cfg.stoch["$type"] == "discrete"
        ):
            raise RuntimeError("Cannot use optimized RSSM")

        self.deter_size = self.cfg.deter_size
        self.num_tokens = cfg.stoch["num_tokens"]
        self.token_size = cfg.stoch["token_size"]
        self.stoch_size = self.num_tokens * self.token_size
        hidden_size = self.cfg.hidden_size

        self._img_in_s = nn.Linear(self.stoch_size, hidden_size)
        self._img_in_a = nn.Linear(self.act_size, hidden_size)
        self._cell = GRUCell(hidden_size, self.deter_size, norm=True)
        self._img_out = nn.Linear(self.deter_size, hidden_size)
        self._img_proj = nn.Linear(hidden_size, self.stoch_size)
        self._obs_out_d = nn.Linear(self.deter_size, hidden_size)
        self._obs_out_o = nn.Linear(obs_size, hidden_size)
        self._obs_proj = nn.Linear(hidden_size, self.stoch_size)

    @torch.jit.ignore
    def initial(self, device, dtype):
        return State(
            deter=torch.zeros([self.deter_size], device=device, dtype=dtype),
            stoch=torch.zeros([self.stoch_size], device=device, dtype=dtype),
        )

    @torch.jit.ignore
    def observe(
        self,
        obs: Tensor,
        act: Tensor,
        state: State,
        sample: bool = True,
    ) -> tuple[State, StateDist, StateDist]:
        assert sample
        obs = over_seq(self._obs_out_o)(obs)
        act = over_seq(self._img_in_a)(act)
        deters, stochs, posts = self._observe(obs, act, state.deter, state.stoch)
        states = State(deters, stochs)
        priors = over_seq(self._img_dist)(deters)
        priors = StateDist(deter=deters, stoch=D.Discrete(logits=priors))
        posts = StateDist(deter=deters, stoch=D.Discrete(logits=posts))
        return states, posts, priors

    @torch.jit.export
    def _observe(self, obs, act, deter, stoch):
        deters: List[Tensor] = []
        stochs: List[Tensor] = []
        posts: List[Tensor] = []

        obs_, act_ = obs.unbind(0), act.unbind(0)
        for t in range(len(obs)):
            deter = self._img_cell(deter, stoch, act_[t])
            post = self._obs_dist(deter, obs_[t])
            stoch = self._discrete_sample(post)
            deters.append(deter)
            stochs.append(stoch)
            posts.append(post)

        return (
            torch.stack(deters),
            torch.stack(stochs),
            torch.stack(posts),
        )

    @torch.jit.ignore
    def obs_step(self, state: State, act: Tensor, next_obs: Tensor, sample=True):
        assert sample
        next_obs = self._obs_out_o(next_obs)
        act = self._img_in_a(act)
        deter, stoch = self._obs_step(state.deter, state.stoch, act, next_obs)
        return State(deter, stoch)

    @torch.jit.export
    def _obs_step(self, deter, stoch, act, next_obs):
        deter = self._img_cell(deter, stoch, act)
        dist = self._obs_dist(deter, next_obs)
        stoch = self._discrete_sample(dist)
        return deter, stoch

    @torch.jit.ignore
    def img_step(self, state: State, act: Tensor, sample=True):
        assert sample
        act = self._img_in_a(act)
        deter, stoch = self._img_step(state.deter, state.stoch, act)
        return State(deter, stoch)

    @torch.jit.export
    def _img_step(self, deter, stoch, act):
        deter = self._img_cell(deter, stoch, act)
        dist = self._img_dist(deter)
        stoch = self._discrete_sample(dist)
        return deter, stoch

    def _img_cell(self, deter, stoch, act):
        x = act + self._img_in_s(stoch)
        x = F.elu(x)
        return self._cell(x, deter)

    def _img_dist(self, deter: Tensor):
        x = self._img_out(deter)
        x = F.elu(x)
        logits = self._img_proj(x)
        logits = logits.reshape(-1, self.num_tokens, self.token_size)
        return logits

    def _obs_dist(self, deter: Tensor, next_obs: Tensor):
        x = self._obs_out_d(deter) + next_obs
        x = F.elu(x)
        logits = self._obs_proj(x)
        logits = logits.reshape(-1, self.num_tokens, self.token_size)
        return logits

    def _discrete_sample(self, logits: Tensor):
        probs = F.softmax(logits, -1)
        eps = 1e-5
        unif = torch.rand_like(probs).clamp(eps, 1 - eps)
        idxes = (logits - (-unif.log()).log()).argmax(-1)
        # idxes = torch.multinomial(probs, 1, True).squeeze(-1)
        sample = F.one_hot(idxes, self.token_size).type_as(probs)
        sample = pass_gradient(sample, probs)
        sample = sample.reshape(-1, self.stoch_size)
        return sample
