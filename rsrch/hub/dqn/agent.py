import torch
from torch import Tensor

from rsrch.rl import agents
from rsrch.utils.eval_ctx import eval_ctx

from .types import QNetwork


class QAgent(agents.Agent):
    def __init__(self, Q: QNetwork):
        self.Q = Q

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self.Q):
            q_values = self.Q(obs.unsqueeze(0)).squeeze(0)
        return torch.argmax(q_values)
