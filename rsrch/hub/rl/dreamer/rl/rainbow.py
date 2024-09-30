from dataclasses import dataclass

from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.hub.rl.dreamer.common.trainer import TrainerBase
from rsrch.rl import gym

from ._dist_q import ValueDist


@dataclass
class DistConfig:
    enabled: bool
    v_min: float
    v_max: float
    num_atoms: int


class Q(nn.Module):
    def __init__(
        self,
        in_features: int,
        act_space: spaces.torch.Discrete,
        hidden_dim: int,
        dist: DistConfig,
    ):
        super().__init__()
        self.in_features = in_features
        self.act_space = act_space
        self.hidden_dim = hidden_dim
        self.dist = dist

        mult = dist.num_atoms if dist.enabled else 1
        self.v_head = self._make_head(mult)
        self.adv_head = self._make_head(mult * act_space.n)

    def _make_head(self, out_features: int):
        return nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_features),
        )

    def forward(self, input: Tensor):
        if self.dist.enabled:
            v: Tensor = self.v_head(input)
            v = v.reshape(len(v), 1, self.dist.num_atoms)
            adv: Tensor = self.adv_head(input)
            adv = adv.reshape(len(adv), self.act_space.n, self.dist.num_atoms)
            logits = v + adv - adv.mean(-2, keepdim=True)
            return ValueDist(
                ind_rv=D.Categorical(logits=logits),
                v_min=self.dist.v_min,
                v_max=self.dist.v_max,
            )
        else:
            adv: Tensor = self.adv_head(input)
            v: Tensor = self.v_head(input)
            return v.flatten(1) + adv - adv.mean(-1, keepdim=True)


class Trainer(TrainerBase):
    ...


class Agent(gym.VecAgent):
    ...
