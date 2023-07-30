import numpy as np
import torch

import rsrch.nn.dist_head as dh
import rsrch.rl.data.transforms as T
from rsrch import nn
from rsrch.exp.board.tb import TensorBoard
from rsrch.exp.board.wandb import Wandb
from rsrch.exp.dir import ExpDir
from rsrch.rl import gym
from rsrch.rl.data.seq import MultiStepBuffer, PaddedSeqBatch
from rsrch.utils import data

from . import core


class A2C(core.A2C):
    def setup_vars(self):
        self.val_every = int(1e3)
        self.train_steps = int(1e6)
        self.val_episodes = 32
        self.gamma = 0.99
        self.log_every = int(1e3)
        self.env_name = "CartPole-v1"
        # self.env_name = "LunarLander-v2"
        self.device = torch.device("cuda")
        self.batch_size = 64
        self.critic_steps = 8
        self.copy_critic_every = 128
        self.gae_lambda = 0.95

    def setup_data(self):
        self.train_env = self._make_env()
        self.val_env = self._make_env()

    def _make_env(self):
        env = gym.make(self.env_name)
        env = gym.wrappers.ToTensor(
            env,
            device=self.device,
            dtype=torch.float32,
        )
        return env

    def setup_models(self):
        obs_space = self.train_env.observation_space
        assert isinstance(obs_space, gym.spaces.TensorBox)
        obs_dim = int(np.prod(obs_space.shape))

        act_space = self.train_env.action_space
        assert isinstance(act_space, gym.spaces.TensorDiscrete)
        act_dim = int(act_space.n)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            dh.Categorical(128, act_dim),
        )

        self.actor = self.actor.to(self.device)

        _make_critic = lambda: nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Flatten(0, -1),
        ).to(self.device)

        self.critic = _make_critic()
        self.target_critic = _make_critic()

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

    def setup_extras(self):
        # self.board = Wandb(project="a2c", step_fn=lambda: self.step)
        self.exp_dir = ExpDir(root="runs/a2c")
        self.board = TensorBoard(
            root_dir=self.exp_dir.path / "board",
            step_fn=lambda: self.step,
        )
