import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from torch.distributions import Categorical
from .utils import eval_ctx
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, obs: Tensor, act: Optional[Tensor] = None):
        batch_size = len(obs)
        obs = obs.reshape(batch_size, -1)
        log_pi = self.net(obs)
        pi = Categorical(logits=log_pi)

        if act is None:
            return pi
        else:
            act = act.reshape(
                -1,
            )
            logp_a = pi.log_prob(act)
            return pi, logp_a

    def act(self, obs: Tensor) -> int:
        with eval_ctx(self):
            pi = self(obs.unsqueeze(0))
            return pi.sample()[0].item()


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, obs: Tensor):
        batch_size = len(obs)
        obs = obs.reshape(batch_size, -1)
        values = self.net(obs)
        return values.reshape(batch_size)


class ActorCritic(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)

        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)

        self.pi = Policy(obs_dim, n_actions)
        self.V = Critic(obs_dim)

    def act(self, obs: torch.Tensor):
        return self.pi.act(obs)


def vpg():
    env = gym.make("LunarLander-v2")
    env_seed = 42
    num_epochs = int(2**13)
    train_steps = np.inf
    train_episodes = 1
    max_ep_length = np.inf
    pi_steps_per_epoch = 1
    V_steps_per_epoch = 1
    val_episodes = 16
    val_epoch_every = 32

    total_train_episodes = 0
    ac = ActorCritic(env)
    pi_opt = torch.optim.Adam(ac.pi.parameters(), lr=1e-3)
    V_opt = torch.optim.Adam(ac.V.parameters(), lr=1e-3)

    device = torch.device("cpu")

    def to_tensor(x, dtype=None):
        return torch.as_tensor(np.array(x), dtype=dtype, device=device)

    def train_epoch():
        nonlocal total_train_episodes

        observations, actions, returns = [], [], []
        ep_observations, ep_actions, ep_rewards = [], [], []

        obs, info = env.reset()
        episode, ep_step, overall_step = 0, 0, 0

        def stop():
            return (
                episode >= train_episodes
                or ep_step >= max_ep_length
                or overall_step >= train_steps
            )

        while not stop():
            action = ac.act(to_tensor(obs, dtype=torch.float32))
            next_obs, reward, term, trunc, info = env.step(action)
            ep_observations.append(obs)
            ep_actions.append(action)
            ep_rewards.append(reward)

            obs = next_obs
            ep_step += 1
            overall_step += 1
            if term or trunc or stop():
                observations.extend(ep_observations)
                actions.extend(ep_actions)

                ep_returns = np.cumsum(ep_rewards[::-1])[::-1]
                returns.extend(ep_returns)

                obs, info = env.reset()
                ep_step = 0
                episode += 1
                total_train_episodes += 1

        observations = to_tensor(observations, dtype=torch.float32)
        actions = to_tensor(actions, dtype=torch.int32)
        returns = to_tensor(returns, dtype=torch.float32)

        values = ac.V(observations)
        advantages = returns - values
        for _ in range(pi_steps_per_epoch):
            pi_opt.zero_grad()
            _, logp_a = ac.pi(observations, actions)
            pi_loss = -(logp_a * advantages).mean()
            pi_loss.backward()
            pi_opt.step()

        for _ in range(V_steps_per_epoch):
            V_opt.zero_grad()
            V_loss = F.mse_loss(ac.V(observations), returns)
            V_loss.backward()
            V_opt.step()

    def val_epoch():
        ep_returns = []
        for _ in range(val_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            while True:
                action = ac.act(to_tensor(obs, dtype=torch.float32))
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                if term or trunc:
                    break
            ep_returns.append(total_reward)

        ep_returns = np.array(ep_returns)
        print(
            f"val epoch: train_episodes={total_train_episodes}, mean={ep_returns.mean():.2f}, std={ep_returns.std():.2f}, min={ep_returns.min():.2f}, max={ep_returns.max():.2f}"
        )

    env.reset(seed=env_seed)
    for epoch in range(1, num_epochs + 1):
        train_epoch()
        if epoch % val_epoch_every == 0:
            val_epoch()
