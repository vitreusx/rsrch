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
        cls = config.get_class(dh, cfg["$type"])
        del cfg["$type"]
        return partial(cls, **cfg)

    def initial(self, device, dtype):
        return State(
            deter=torch.zeros([self.deter_size], device=device, dtype=dtype),
            stoch=torch.zeros([self.stoch_size], device=device, dtype=dtype),
        )

    def observe(
        self,
        obs: Tensor,
        act: Tensor,
        state: State,
        sample: bool = True,
    ) -> tuple[StateDist, StateDist]:
        posts, priors = [], []
        for t in range(obs.shape[0]):
            deter = self._img_cell(state, act[t])
            prior = self._img_dist(deter)
            priors.append(StateDist(deter=deter, stoch=prior))
            post = self._obs_dist(deter, obs[t])
            posts.append(StateDist(deter=deter, stoch=post))
            stoch = post.rsample() if sample else post.mode
            state = State(deter, stoch)

        posts = torch.stack(posts)
        priors = torch.stack(priors)
        return posts, priors

    def obs_step(
        self,
        state: State,
        act: Tensor,
        next_obs: Tensor,
        sample: bool = True,
    ) -> State:
        deter = self._img_cell(state, act)
        dist = self._obs_dist(deter, next_obs)
        stoch = dist.rsample() if sample else dist.mode
        return State(deter, stoch)

    def img_step(
        self,
        state: State,
        act: Tensor,
        sample: bool = True,
    ) -> State:
        deter = self._img_cell(state, act)
        dist = self._img_dist(deter)
        stoch = dist.rsample() if sample else dist.mode
        return State(deter, stoch)

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
