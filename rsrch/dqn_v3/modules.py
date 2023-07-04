import torch
from torch import Tensor
from typing import List
import torch.nn as nn
import gymnasium as gym


class MLP(nn.Sequential):
    def __init__(
        self, num_features: List[int], norm_layer=nn.LayerNorm, act_layer=nn.ReLU
    ):
        layers = []
        seq = enumerate(zip(num_features[:-1], num_features[1:]))
        final_layer_idx = len(num_features) - 1
        for layer_idx, (in_features, out_features) in seq:
            if layer_idx > 0:
                layers.append(norm_layer(in_features))
                layers.append(act_layer())
            bias = layer_idx == final_layer_idx
            layers.append(nn.Linear(in_features, out_features, bias=bias))

        super().__init__(*layers)


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


class ObsEncoder(nn.Module):
    def __init__(self, obs_space: gym.Space, enc_dim: int):
        super().__init__()
        self.out_features = enc_dim

        if isinstance(obs_space, gym.spaces.Box):
            obs_shape = torch.Size(obs_space.shape)
            if len(obs_shape) >= 3:
                self.enc = VisualEncoder(obs_shape)
                self.enc = nn.Sequential(
                    self.enc, nn.Linear(self.enc.out_features, enc_dim)
                )
            else:
                self.enc = MLP([obs_shape.numel(), 256, enc_dim])
        else:
            raise NotImplementedError()

    def forward(self, obs: Tensor) -> Tensor:
        return self.enc(obs)


class ActionEncoder(nn.Module):
    def __init__(self, act_space: gym.Space, enc_dim: int):
        super().__init__()
        self.out_features = enc_dim

        if isinstance(act_space, gym.spaces.Discrete):
            self.enc = nn.Embedding(int(act_space.n), enc_dim)
        elif isinstance(act_space, gym.spaces.Box):
            shape = torch.Size(act_space.shape)
            self.enc = MLP([shape.numel(), 256, enc_dim])
        else:
            raise NotImplementedError()

    def forward(self, act: Tensor) -> Tensor:
        return self.enc(act)
