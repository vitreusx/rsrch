import torch
import torch.nn.functional as F
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch.rl import gym
from rsrch.types import Tensorlike


class State(Tensorlike):
    def __init__(self, deter: Tensor, stoch: Tensor):
        Tensorlike.__init__(self, stoch.shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)


class StateDist(Tensorlike, D.Distribution):
    def __init__(self, deter: Tensor, stoch: D.Distribution):
        Tensorlike.__init__(self, stoch.batch_shape)
        self.deter = self.register("deter", deter)
        self.stoch = self.register("stoch", stoch)

    def rsample(self, sample_shape=()):
        return State(self.deter, self.stoch.rsample(sample_shape))

    def sample(self, sample_shape=()):
        return State(self.deter, self.stoch.sample(sample_shape))


class StateToTensor(nn.Module):
    def forward(self, state: State) -> Tensor:
        return torch.cat([state.deter, state.stoch], -1)


class PredCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.deter: nn.RNNCellBase
        self.deter_norm = nn.LayerNorm([self._deter.hidden_size])
        self.stoch: nn.Module

    def forward(self, state: State, input: Tensor) -> StateDist:
        input = torch.cat([input, state.stoch], 1)
        next_deter = self.deter(state.deter, input)
        next_deter = self.deter_norm(next_deter)
        next_stoch_dist = self.stoch(next_deter)
        return StateDist(next_deter, next_stoch_dist)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.deter_dim: int
        self.stoch_dim: int
        self.obs_dim: int
        self.act_dim: int
        self._init: nn.RNNBase
        self._belief: nn.RNNBase
        self.proj: nn.Module
        self.pred: PredCell

    @property
    def device(self):
        return next(self.parameters()).device

    def init(self, obs: Tensor) -> Tensor:
        _, init_h = self._init(obs[None].contiguous())
        return init_h

    def beliefs(
        self,
        act_seq: Tensor,
        obs_seq: Tensor,
        init_h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        is_none = init_h is None
        if is_none:
            init_h = self.init(obs_seq[0])
            obs_seq = obs_seq[1:]
        input = torch.cat([act_seq, obs_seq], 1)
        out, next_h = self._belief(input.contiguous(), init_h.contiguous())
        if is_none:
            out = torch.cat([init_h[-1][None], out], 0)
        return out, next_h

    def update(self, cur_h: Tensor, act: Tensor, next_obs: Tensor) -> Tensor:
        input = torch.cat([act, next_obs], 1)[None]
        _, next_h = self._belief(input.contiguous(), cur_h.contiguous())
        return next_h


def state_dist_div(p: StateDist, q: StateDist):
    return D.kl_divergence(p.stoch, q.stoch)


class VecAgent(gym.VecAgent):
    def __init__(
        self,
        actor: nn.Module,
        wm: RSSM,
        obs_enc: nn.Module,
        act_enc: nn.Module,
        env_f,
        num_envs: int,
    ):
        super().__init__()
        self.actor = actor
        self.wm = wm
        self.env_f = env_f
        self.obs_enc, self.act_enc = obs_enc, act_enc

        self._obs = torch.empty((num_envs, wm.obs_dim), device=wm.device)
        self._state = State(
            deter=torch.empty((num_envs, wm.deter_dim), device=wm.device),
            stoch=torch.empty((num_envs, wm.stoch_dim), device=wm.device),
        )

    def reset(self, idxes, obs, info):
        obs = self.env_f.move_obs(obs)
        obs = self.obs_enc(obs)
        self._obs[idxes] = obs
        self._state[idxes] = self.wm.init(obs)

    def policy(self, obs):
        act = self.actor(self._state)
        return self.env_f.move_act(act, to="env")

    def step(self, act):
        act = self.env_f.move_act(act, to="net")
        self._act = self.act_enc(act)

    def observe(self, idxes, next_obs, term, trunc, info):
        next_obs = self.env_f.move_obs(next_obs)
        next_obs = self.obs_enc(next_obs)
        self._obs[idxes] = next_obs
        self._state[idxes] = self.wm.update(
            self._state[idxes], self._act[idxes], next_obs
        )
