from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.nn import dist_head as dh
from rsrch.types import Tensorlike

from . import config, nets
from .nets import *


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
    def __init__(self, cfg: config.RSSM, obs_size: int, act_size: int):
        super().__init__()
        self.cfg = cfg
        self.obs_size = obs_size
        self.act_size = act_size

        self.deter_size = self.cfg.deter_size
        hidden_size = self.cfg.hidden_size

        norm_layer = NormLayer1d(self.cfg.norm)
        act_layer = ActLayer(self.cfg.act)
        stoch_layer = self._make_stoch(self.cfg.stoch)

        stoch_test = stoch_layer(hidden_size)
        with safe_mode(stoch_test):
            x = torch.empty(hidden_size)
            self.stoch_size = stoch_test(x[None]).mode.shape[-1]

        self._img_in = nn.Sequential(
            nn.Linear(self.stoch_size + act_size, hidden_size),
            norm_layer(hidden_size),
            act_layer(),
        )

        self._cell = GRUCell(
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

    def _make_stoch(self, cfg: dict):
        cfg = {**cfg}
        # if $type is, for example, "normal", then the class is dh.Normal
        cls = getattr(dh, cfg["$type"].capitalize())
        del cfg["$type"]
        return partial(cls, **cfg)

    def initial(self, device, dtype):
        return State(
            deter=torch.zeros([self.deter_size], device=device, dtype=dtype),
            stoch=torch.zeros([self.stoch_size], device=device, dtype=dtype),
        )

    @torch.compile
    def observe(
        self,
        obs: Tensor,
        act: Tensor,
        state: State,
        is_first: Tensor,
        sample: bool = True,
    ) -> tuple[StateDist, StateDist]:
        seq_len, batch_size = obs.shape[:2]

        posts, priors = [], []
        for t in range(seq_len):
            is_first_ = is_first if t == 0 else None
            post, prior = self.obs_step(state, act[t], obs[t], is_first_)
            posts.append(post)
            priors.append(prior)
            state = post.rsample() if sample else post.mode

        return torch.stack(posts), torch.stack(priors)

    def imagine(
        self,
        act: Tensor,
        state: State | None = None,
        sample: bool = True,
    ) -> StateDist:
        seq_len, batch_size = act.shape[:2]

        if state is None:
            state = self.initial(act.device, act.dtype)
            state = state.expand(batch_size, *state.shape)

        priors = []
        for t in range(seq_len):
            prior = self.img_step(state, act[t])
            priors.append(prior)
            state = prior.rsample() if sample else prior.mode

        return torch.stack(priors)

    def obs_step(
        self,
        prev_state: State,
        act: Tensor,
        next_obs: Tensor,
        is_first: Tensor | None,
    ) -> tuple[StateDist, StateDist]:
        if is_first is not None:
            prev_state = prev_state.clone()
            prev_state[is_first].zero_()
            act = act.clone()
            act[is_first].zero_()

        prior = self.img_step(prev_state, act)

        x = torch.cat((prior.deter, next_obs), -1)
        stoch = self._obs_out(x)
        post = StateDist(deter=prior.deter, stoch=stoch)

        return post, prior

    def img_step(self, prev_state: State, act: Tensor) -> StateDist:
        x = torch.cat((prev_state.stoch, act), -1)
        x = self._img_in(x)
        deter = self._cell(x, prev_state.deter)

        ens_idx = np.random.randint(len(self._img_out))
        stoch = self._img_out[ens_idx](deter)

        return StateDist(deter=deter, stoch=stoch)
