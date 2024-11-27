from dataclasses import dataclass

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from dreamerx.common.utils import null_ctx
from dreamerx.wm import WorldModel
from rsrch import spaces
from rsrch.nn.utils import over_seq
from rsrch.rl import gym


@dataclass
class Config:
    lookahead: int
    num_samples: int
    num_elites: int
    num_iters: int


class Agent(gym.vector.agents.Markov):
    def __init__(
        self,
        cfg: Config,
        wm: WorldModel,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(wm.state_space, wm.act_space)
        self.cfg = cfg
        self.wm = wm
        self.device = next(self.wm.parameters()).device
        self.act_space: spaces.torch.Tensor = self.wm.act_space
        if not isinstance(self.act_space, (spaces.torch.OneHot,)):
            raise ValueError(self.act_space)
        self.compute_dtype = compute_dtype

    def autocast(self):
        if self.compute_dtype is None:
            return null_ctx()
        else:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
            )

    def get_policy(self, state: Tensor):
        with self.autocast():
            with torch.no_grad():
                L, N, S, E = (
                    self.cfg.lookahead,
                    state.shape[0],
                    self.cfg.num_samples,
                    self.cfg.num_elites,
                )

                act_shape = (L, N, *self.act_space.shape)
                kw = dict(dtype=state.dtype, device=state.device)
                if isinstance(self.act_space, spaces.torch.OneHot):
                    act_dist = D.OneHot(logits=torch.zeros(act_shape, **kw))

                h_0 = state[None].expand(S, *state.shape)  # [S, N, *A]
                h_0 = h_0.flatten(0, 1)

                for _ in range(self.cfg.num_iters):
                    acts = act_dist.sample((self.cfg.num_samples,))  # [S, L, N, *A]
                    act_seq = acts.moveaxis(1, 0).flatten(1, 2)

                    out, hx = self.wm.imagine(act_seq, h_0)
                    states: Tensor = out.states  # [L, S * N, *St]

                    rewards = over_seq(self.wm.reward_dec)(states).mean  # [L, S * N]
                    rewards = rewards.reshape(L, S, N)
                    totals = rewards.sum(0)  # [S, N]

                    _, elite_idx = torch.topk(totals, E, dim=0)  # [E, N]

                    elite_idx = elite_idx.reshape(
                        E, 1, N, *(1 for _ in self.act_space.shape)
                    )
                    elite_idx = elite_idx.expand(E, L, N, *self.act_space.shape)
                    elite_act = torch.gather(acts, 0, elite_idx)  # [E, L, N, *A]

                    if isinstance(self.act_space, spaces.torch.OneHot):
                        act_dist = D.OneHot(probs=elite_act.mean(0))

                act_dist = act_dist[0]
                return act_dist.sample()
