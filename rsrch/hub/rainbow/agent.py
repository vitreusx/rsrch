import numpy as np
import torch
from torch import Tensor
from rsrch.rl import gym
from torch import nn
from .distr_q import ValueDist
from .env import Loader


class QAgent(gym.vector.Agent):
    def __init__(self, q: nn.Module, loader: Loader):
        self.q = q
        self._device = next(q.parameters()).device
        self._loader = loader

    @torch.inference_mode()
    def policy(self, obs: np.ndarray):
        obs = self._loader.conv_obs(obs).to(self._device)
        q_values: ValueDist | Tensor = self.q(obs)

        if isinstance(q_values, ValueDist):
            act = q_values.mean.argmax(-1)
        else:
            act = q_values.argmax(-1)

        return act.cpu().numpy()
