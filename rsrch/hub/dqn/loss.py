from typing import Optional

import torch
import torch.nn as nn

from rsrch.rl.data import StepBatch
from rsrch.utils.eval_ctx import eval_ctx

from .types import QNetwork


class DQNLoss(nn.Module):
    def __init__(self, Q: QNetwork, gamma: float, target_Q: Optional[QNetwork] = None):
        super().__init__()
        self.Q = Q
        self.gamma = gamma
        self.target_Q = target_Q or Q

    def forward(self, batch: StepBatch):
        q_values = self.Q(batch.obs)
        preds = q_values.gather(1, batch.act.unsqueeze(1)).squeeze(1)

        with eval_ctx(self.target_Q):
            next_q = self.target_Q(batch.next_obs)
            next_preds, _ = next_q.max(1)
            gamma_t = self.gamma * (1.0 - batch.term.float())
            target = batch.reward + gamma_t * next_preds

        return nn.functional.mse_loss(preds, target)
