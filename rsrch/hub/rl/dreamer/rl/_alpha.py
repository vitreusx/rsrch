from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces


@dataclass
class Config:
    adaptive: bool
    init_value: float = 1e-8
    amp_rate: float = 0.0
    decay_rate: float = 0.0
    min_ent: float | Literal["auto"] = "auto"


def auto_min_ent(act_space: spaces.torch.Tensor) -> float:
    if isinstance(act_space, spaces.torch.Discrete):
        # Target: analogue of an eps-greedy policy, with eps = 0.5
        eps, N = 0.5, act_space.n
        q, p = eps / N, 1.0 - eps + eps / N
        probs = torch.tensor([p, *(q for _ in range(N))])
        dist = D.Categorical(probs=probs)
        return dist.entropy().item()
    elif isinstance(act_space, spaces.torch.Box):
        # Target: normal distribution with scale ratio of 5e-2 of the extent
        # of the action space
        scale = 7.5e-2 * (act_space.high - act_space.low)
        dist = D.Normal(0, scale, len(act_space.shape))
        return dist.entropy().item()
    else:
        raise ValueError(type(act_space))


class Alpha(nn.Module):
    """Alpha parameter for entropy regularization."""

    def __init__(self, cfg: Config, act_space: spaces.torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.adaptive = cfg.adaptive

        if self.adaptive:
            self.value = self.cfg.init_value
            if cfg.min_ent == "auto":
                self.min_ent = auto_min_ent(act_space)
            else:
                self.min_ent = cfg.min_ent
        else:
            self.value = cfg.init_value

    def opt_step(self, entropy: Tensor):
        """Optimize alpha value based on current entropy estimates.
        :param entropy: Tensor of shape (N,) containing entropy estimates.
        :param metrics: Optional dict. If passed, auxiliary metrics are computed
        and inserted into the dict.
        :return: The loss value (or None if alpha is not adaptive).
        """

        if self.adaptive:
            mean_ent = entropy.mean().item()
            if mean_ent > self.min_ent:
                self.value /= 1.0 + self.cfg.decay_rate
            else:
                self.value *= 1.0 + self.cfg.amp_rate

            with torch.no_grad():
                metrics = {}
                metrics["value"] = self.value
                metrics["entropy"] = mean_ent

            return metrics

    def __float__(self):
        return self.value
