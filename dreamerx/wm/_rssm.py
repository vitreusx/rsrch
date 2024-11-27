from collections import namedtuple
from dataclasses import dataclass
from functools import cached_property, partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dh
from rsrch.nn.utils import over_seq, pass_gradient, safe_mode
from rsrch.types import Tensorlike

from ..common import nets
from ..common.utils import tf_init


@dataclass
class Config:
    ensemble: int
    deter_size: int
    stoch: dict
    act: nets.ActType
    norm: nets.NormType
    hidden_size: int
    jit: bool


Output = namedtuple("Output", ("states", "post", "prior"))


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        super().__init__(deter.shape[:-1])
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def zero_(self):
        self.deter.zero_()
        self.stoch.zero_()
        return self

    @cached_property
    def _as_tensor(self):
        return torch.cat((self.deter, self.stoch), -1)

    def as_tensor(self):
        return self._as_tensor


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


class GenericRSSM(nn.Module):
    def __init__(self, cfg: Config, obs_size: int, act_size: int):
        super().__init__()
        self.cfg = cfg
        self.obs_size = obs_size
        self.act_size = act_size

        self.deter_size = self.cfg.deter_size
        hidden_size = self.cfg.hidden_size

        norm_layer = partial(nets.NormLayer1d, self.cfg.norm)
        act_layer = partial(nets.ActLayer, self.cfg.act)
        stoch_layer = self._stoch_ctor(self.cfg.stoch)

        stoch_test = stoch_layer(hidden_size)
        with safe_mode(stoch_test):
            x = torch.empty(hidden_size)
            self.stoch_size = stoch_test(x[None]).mode.shape[-1]

        self._img_in = nn.Sequential(
            nn.Linear(self.stoch_size + act_size, hidden_size),
            norm_layer(hidden_size),
            act_layer(),
        )

        self._cell = nets.GRUCellLN(
            input_size=hidden_size,
            hidden_size=self.deter_size,
            norm=True,
        )

        self._img_out = nn.ModuleList()
        for _ in range(self.cfg.ensemble):
            self._img_out.append(
                nn.Sequential(
                    nn.Linear(self.deter_size, hidden_size),
                    norm_layer(hidden_size),
                    act_layer(),
                    stoch_layer(hidden_size),
                ),
            )

        self._obs_out = nn.Sequential(
            nn.Linear(self.deter_size + obs_size, hidden_size),
            norm_layer(hidden_size),
            act_layer(),
            stoch_layer(hidden_size),
        )

        self.register_buffer("deter0", torch.zeros(self.deter_size))
        self.register_buffer("stoch0", torch.zeros(self.stoch_size))

        self.apply(tf_init)

    def _stoch_ctor(self, cfg: dict):
        cfg = {**cfg}
        typ = cfg["type"]
        del cfg["type"]

        def ctor(in_features: int):
            nonlocal cfg
            layer_ctor = partial(nn.Linear, in_features)
            if typ == "discrete":
                space = spaces.torch.TokenSeq(**cfg)
                return dh.Discrete(layer_ctor, space, init="tf")
            elif typ == "normal":
                space = spaces.torch.Tensor((cfg["size"],))
                del cfg["size"]
                cfg = {**cfg, "init": "tf"}
                return dh.Normal(layer_ctor, space, **cfg)

        return ctor

    def initial(self):
        return State(self.deter0, self.stoch0)

    def forward(
        self,
        input: tuple[Tensor, Tensor],
        h_0: State,
        sample: bool = True,
    ):
        obs, act = input
        states, posts, priors = [], [], []

        state = h_0
        for t in range(obs.shape[0]):
            deter = self._img_cell(state, act[t])
            prior = self._img_dist(deter)
            priors.append(StateDist(deter=deter, stoch=prior))
            post = self._obs_dist(deter, obs[t])
            posts.append(StateDist(deter=deter, stoch=post))
            stoch = post.rsample() if sample else post.mode
            state = State(deter, stoch)
            states.append(state)

        states = torch.stack(states)
        posts = torch.stack(posts)
        priors = torch.stack(priors)
        return Output(states, posts, priors), states[-1]

    def obs_step(
        self,
        state: State,
        act: Tensor,
        next_obs: Tensor,
    ) -> StateDist:
        deter = self._img_cell(state, act)
        dist = self._obs_dist(deter, next_obs)
        return StateDist(deter=deter, stoch=dist)

    def img_step(
        self,
        state: State,
        act: Tensor,
    ) -> StateDist:
        deter = self._img_cell(state, act)
        dist = self._img_dist(deter)
        return StateDist(deter=deter, stoch=dist)

    def _img_cell(self, state: State, act: Tensor):
        x = torch.cat((state.stoch, act), -1)
        x = self._img_in(x)
        return self._cell(x, state.deter)

    def _img_dist(self, deter: Tensor):
        ens_idx = np.random.randint(len(self._img_out))
        return self._img_out[ens_idx](deter)

    def _obs_dist(self, deter: Tensor, next_obs: Tensor):
        x = torch.cat((deter, next_obs), -1)
        return self._obs_out(x)


class OptRSSM(nn.Module):
    __constants__ = [
        "obs_size",
        "act_size",
        "deter_size",
        "num_tokens",
        "vocab_size",
        "stoch_size",
    ]

    def __init__(self, cfg: Config, obs_size: int, act_size: int):
        super().__init__()
        self.cfg = cfg
        self.obs_size = obs_size
        self.act_size = act_size

        if not (
            cfg.ensemble == 1
            and cfg.norm == "none"
            and cfg.act == "elu"
            and cfg.stoch["type"] == "discrete"
        ):
            raise RuntimeError("Cannot use optimized RSSM")

        self.deter_size = self.cfg.deter_size
        self.num_tokens = cfg.stoch["num_tokens"]
        self.vocab_size = cfg.stoch["vocab_size"]
        self.stoch_size = self.num_tokens * self.vocab_size
        hidden_size = self.cfg.hidden_size

        self._img_in_s = nn.Linear(self.stoch_size, hidden_size)
        self._img_in_a = nn.Linear(self.act_size, hidden_size, bias=False)
        self._cell = nets.GRUCellLN(hidden_size, self.deter_size, norm=True)
        self._img_out = nn.Linear(self.deter_size, hidden_size)
        self._img_proj = nn.Linear(hidden_size, self.stoch_size)
        self._obs_out_d = nn.Linear(self.deter_size, hidden_size)
        self._obs_out_o = nn.Linear(self.obs_size, hidden_size, bias=False)
        self._obs_proj = nn.Linear(hidden_size, self.stoch_size)

        self.register_buffer("deter0", torch.zeros(self.deter_size))
        self.register_buffer("stoch0", torch.zeros(self.stoch_size))

        self.apply(tf_init)

        # Explanation for the stuff below:
        # The original implementation does img_in(concat(stoch, act)). We wish to split it into img_in_s(stoch) + img_in_a(act), so that we (1) avoid concat, (2) can run img_in_a(act) only once, since it doesn't depend on the sequence position. However, naive replacement (with default initialization of weights and biases) doesn't yield the same result. The remedy is to create temporary img_in layer and copy the respective parameters from it to img_in_s and img_in_a.

        img_in = nn.Linear(self.stoch_size + self.act_size, hidden_size)
        img_in.apply(tf_init)
        W_s, W_a = img_in.weight.split_with_sizes(
            [self.stoch_size, self.act_size],
            dim=1,
        )
        self._img_in_s.weight.data.copy_(W_s)
        self._img_in_a.weight.data.copy_(W_a)

        # Here it's the same as above, but with obs_out.

        obs_out = nn.Linear(self.deter_size + self.obs_size, hidden_size)
        obs_out.apply(tf_init)
        W_d, W_o = obs_out.weight.split_with_sizes(
            [self.deter_size, self.obs_size],
            dim=1,
        )
        self._obs_out_d.weight.data.copy_(W_d)
        self._obs_out_o.weight.data.copy_(W_o)

    @torch.jit.ignore
    def initial(self):
        return State(self.deter0, self.stoch0)

    @torch.jit.ignore
    def forward(
        self,
        input: tuple[Tensor, Tensor],
        h_0: State,
        sample: bool = True,
    ):
        return self.observe(input, h_0, sample)

    @torch.jit.ignore
    def observe(
        self,
        input: tuple[Tensor, Tensor],
        h_0: State,
        sample: bool = True,
    ):
        assert sample
        obs, act = input
        obs = over_seq(self._obs_out_o)(obs)
        act = over_seq(self._img_in_a)(act)
        deters, stochs, posts = self._observe(obs, act, h_0.deter, h_0.stoch)
        states = State(deters, stochs)
        priors = over_seq(self._img_dist)(deters)
        priors = StateDist(deter=deters, stoch=D.Discrete(logits=priors))
        posts = StateDist(deter=deters, stoch=D.Discrete(logits=posts))
        return Output(states, posts, priors), states[-1]

    @torch.jit.ignore
    def imagine(
        self,
        act: Tensor,
        h_0: State,
        sample: bool = True,
    ):
        assert sample
        act = over_seq(self._img_in_a)(act)
        deters, stochs, priors = self._imagine(act, h_0.deter, h_0.stoch)
        states = State(deters, stochs)
        priors = StateDist(deter=deters, stoch=D.Discrete(logits=priors))
        return Output(states, None, priors), states[-1]

    @torch.jit.ignore
    def obs_step(self, state: State, act: Tensor, next_obs: Tensor):
        next_obs = self._obs_out_o(next_obs)
        act = self._img_in_a(act)
        deter, logits = self._obs_step(state.deter, state.stoch, act, next_obs)
        stoch = D.Discrete(logits=logits)
        return StateDist(deter=deter, stoch=stoch)

    @torch.jit.ignore
    def img_step(self, state: State, act: Tensor):
        act = self._img_in_a(act)
        deter, logits = self._img_step(state.deter, state.stoch, act)
        stoch = D.Discrete(logits=logits)
        return StateDist(deter=deter, stoch=stoch)

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

    @torch.jit.export
    def _imagine(self, act, deter, stoch):
        deters: List[Tensor] = []
        stochs: List[Tensor] = []
        priors: List[Tensor] = []

        act_ = act.unbind(0)
        for t in range(len(act)):
            deter = self._img_cell(deter, stoch, act_[t])
            prior = self._img_dist(deter)
            stoch = self._discrete_sample(prior)
            deters.append(deter)
            stochs.append(stoch)
            priors.append(prior)

        return torch.stack(deters), torch.stack(stochs), torch.stack(priors)

    @torch.jit.export
    def _obs_step(self, deter, stoch, act, next_obs):
        deter = self._img_cell(deter, stoch, act)
        dist = self._obs_dist(deter, next_obs)
        return deter, dist

    @torch.jit.export
    def _img_step(self, deter, stoch, act):
        deter = self._img_cell(deter, stoch, act)
        dist = self._img_dist(deter)
        return deter, dist

    def _img_cell(self, deter, stoch, act):
        x = act + self._img_in_s(stoch)
        x = F.elu(x)
        return self._cell(x, deter)

    def _img_dist(self, deter: Tensor):
        x = self._img_out(deter)
        x = F.elu(x)
        logits = self._img_proj(x)
        logits = logits.reshape(-1, self.num_tokens, self.vocab_size)
        return logits

    def _obs_dist(self, deter: Tensor, next_obs: Tensor):
        x = self._obs_out_d(deter) + next_obs
        x = F.elu(x)
        logits = self._obs_proj(x)
        logits = logits.reshape(-1, self.num_tokens, self.vocab_size)
        return logits

    def _discrete_sample(self, logits: Tensor):
        logits = logits.to(torch.float32)
        probs = F.softmax(logits, -1)
        eps = self._get_eps(probs)
        unif = torch.rand_like(probs).clamp(eps, 1.0 - eps)
        idxes = (logits - (-unif.log()).log()).argmax(-1)
        # idxes = torch.multinomial(probs, 1, True).squeeze(-1)
        sample = F.one_hot(idxes, self.vocab_size).type_as(probs)
        sample = pass_gradient(sample, probs)
        sample = sample.reshape(-1, self.stoch_size)
        return sample

    def _get_eps(
        self,
        x: Tensor,
        eps16: float = torch.finfo(torch.float16).eps,
        eps32: float = torch.finfo(torch.float32).eps,
        eps64: float = torch.finfo(torch.float64).eps,
    ) -> float:
        """https://github.com/pytorch/pytorch/issues/25661#issuecomment-845419189"""
        if x.dtype == torch.float16:
            return eps16
        elif x.dtype == torch.float32:
            return eps32
        elif x.dtype == torch.float64:
            return eps64
        else:
            raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")


def RSSM(cfg: Config, obs_size: int, act_size: int) -> OptRSSM | GenericRSSM:
    if cfg.jit:
        module = OptRSSM(cfg, obs_size, act_size)
        module: OptRSSM = torch.jit.script(module)
    else:
        module = GenericRSSM(cfg, obs_size, act_size)
    return module


class AsTensor(nn.Module):
    def forward(self, state: State):
        return state.as_tensor()
