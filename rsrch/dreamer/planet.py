from dataclasses import dataclass

import torch
import torch.distributions as dist
import torch.nn as nn
from torch import Tensor

from rsrch.nn.fc import FullyConnected
from rsrch.nn.normal import NormalLinear
from rsrch.rl.spec import EnvSpec

from .rssm import RSSMCell, RSSMState, RSSMStateDist


class VisualEncoder(nn.Module):
    def __init__(self, obs_shape: torch.Size, kernel_size=3, hidden_dim=32):
        super().__init__()
        self.obs_shape = obs_shape
        self.kernel_size = k = kernel_size
        self.hidden_dim = h = hidden_dim

        c, W, H = self.obs_shape
        assert W % 16 == 0 and H % 16 == 0, "image resolution should be divisible by 16"
        assert k % 2 == 1, "kernel_size should be an odd number"
        p = k // 2

        final_size = torch.Size([H // 16, W // 16, 8 * h])
        self.out_features = final_size.numel()

        self.main = nn.Sequential(
            nn.Conv2d(c, h, k, 1, p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(h, 2 * h, k, 1, p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * h, 4 * h, k, 1, p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * h, 8 * h, k, 1, p),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class VisualDecoder(nn.Module):
    def __init__(self, obs_shape: torch.Size, kernel_size=3, hidden_dim=32):
        super().__init__()
        self.obs_shape = obs_shape
        self.kernel_size = k = kernel_size
        self.hidden_dim = h = hidden_dim

        c, self.W, self.H = self.obs_shape
        p = k // 2

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(8 * h, 4 * h, k, 1, p),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(4 * h, 2 * h, k, 1, p),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * h, h, k, 1, p),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(h, c, k, 1, p),
        )

    def forward(self, x: Tensor) -> dist.Distribution:
        x = x.reshape(len(x), -1, self.W // 16, self.H // 16)
        obs_dist = dist.Normal(self.main(x), 1.0)
        # By default, we'd get a single normal distribution over [B, C, H, W],
        # whereas we actually want B independent normal distributions over
        # [C, H, W] each. This is what Independent is for.
        obs_dist = dist.Independent(obs_dist, 3)
        return obs_dist


class PlaNetRSSM(nn.Module):
    def __init__(
        self,
        spec: EnvSpec,
        deter_dim: int,
        stoch_dim: int,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        obs_shape = spec.observation_space.shape
        act_shape = spec.action_space.shape

        self.vis_encoder = VisualEncoder(obs_shape)
        repr_model_input = self.vis_encoder.out_features + act_shape.numel()
        self.repr_cell = RSSMCell(repr_model_input, deter_dim, stoch_dim)
        trans_model_input = spec.action_space.shape.numel()
        self.trans_cell = RSSMCell(trans_model_input, deter_dim, stoch_dim)
        self.vis_decoder = VisualDecoder(deter_dim + stoch_dim, obs_shape)
        self.reward_net = nn.Sequential(
            FullyConnected([deter_dim + stoch_dim, 256], final_layer="act"),
            NormalLinear(256, 1),
        )

    def repr_model(self, h: RSSMState, obs: Tensor, act: Tensor) -> RSSMStateDist:
        obs_z = self.vis_encoder(obs)
        act_z = act.reshape(len(act), -1)
        repr_x = torch.cat([obs_z, act_z], dim=1)
        return self.repr_cell(h, repr_x)

    def trans_model(self, h: RSSMState, act: Tensor) -> RSSMStateDist:
        act_z = act.reshape(len(act), -1)
        trans_x = act_z
        return self.trans_cell(h, trans_x)

    def obs_model(self, h: RSSMState) -> dist.Distribution:
        state = torch.stack([h.deter, h.stoch], dim=1)
        return self.vis_decoder(state)

    def reward_model(self, h: RSSMState) -> dist.Distribution:
        return self.reward_net(h.as_tensor())


class CEMPlanner:
    def __init__(
        self,
        horizon: int,
        optim_iters: int,
        iter_pop: int,
        iter_top_k: int,
        env_spec: EnvSpec,
    ):
        self.horizon = self.H = horizon
        self.optim_iters = self.I = optim_iters
        self.iter_pop = self.J = iter_pop
        self.iter_top_k = self.K = iter_top_k
        self.spec = env_spec

    def next_action(self, cur_state_dist: RSSMStateDist, planet: PlaNetRSSM):
        # Initialize action sequence distribution
        mean = torch.zeros((self.H, *self.act_shape))
        std = torch.ones((self.H, *self.act_shape))
        action_seq_dist = dist.Normal(mean, std)

        rewards = torch.zeros(self.J, requires_grad=False)

        for iter in range(self.I):
            # Sample actions from the distribution
            actions = action_seq_dist.sample_n(self.J)

            # Execute actions and predict the rewards
            rewards.zero_()
            state = cur_state_dist.sample_n(self.J)
            rewards += planet.reward_model(state).mean
            for step_idx in range(self.H):
                state = planet.trans_model(state, actions[:, step_idx]).sample()
                rewards += planet.reward_model(state).mean

            # Select the best sequences and update the distribution
            best_idxes = torch.argmax(rewards)[: self.K]
            best_actions = actions[best_idxes]
            mean, std = best_actions.mean(0), best_actions.var(0)
            action_seq_dist = dist.Normal(mean, std)

        # Return mean first action value
        return mean[0]
