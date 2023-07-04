import numpy as np
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.transform_reward import TransformReward
from gymnasium.wrappers.transform_observation import TransformObservation
from dataclasses import dataclass
import torch
from typing import Optional
import torch.nn as nn
from .utils import eval_ctx
import torch.nn.functional as F


@dataclass
class Transition:
    obs: np.ndarray
    act: int
    reward: float
    next_obs: np.ndarray
    done: bool


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    act: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def to(self, device: torch.device):
        return TransitionBatch(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            next_obs=self.next_obs.to(device),
            done=self.done.to(device),
        )


class ReplayBuffer:
    def __init__(self, env: gym.Env, capacity: int):
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_type = env.observation_space.dtype
        self.obs = np.empty((capacity, *env.observation_space.shape), dtype=obs_type)

        assert isinstance(env.action_space, gym.spaces.Discrete)
        act_type = env.action_space.dtype
        self.act = np.empty((capacity,), dtype=act_type)

        self.reward = np.empty((capacity,), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.done = np.empty((capacity,), dtype=np.float32)

        self.obs = torch.from_numpy(self.obs)
        self.act = torch.from_numpy(self.act)
        self.reward = torch.from_numpy(self.reward)
        self.next_obs = torch.from_numpy(self.next_obs)
        self.done = torch.from_numpy(self.done)

        self.capacity = capacity
        self.size = 0
        self.ins_idx = 0

    def add(self, tr: Transition):
        self.obs[self.ins_idx] = torch.from_numpy(tr.obs)
        self.act[self.ins_idx] = tr.act
        self.reward[self.ins_idx] = tr.reward
        self.next_obs[self.ins_idx] = torch.from_numpy(tr.next_obs)
        self.done[self.ins_idx] = float(tr.done)
        self.ins_idx = (self.ins_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idxes = torch.randint(high=self.size, size=(n,))

        return TransitionBatch(
            obs=self.obs[idxes],
            act=self.act[idxes],
            reward=self.reward[idxes],
            next_obs=self.next_obs[idxes],
            done=self.done[idxes],
        )


class QNetwork(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape[1:] == (84, 84)
        in_channels = env.observation_space.shape[0]

        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.num_actions = int(env.action_space.n)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.num_actions),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs.float())


class GreedyAgent:
    def __init__(self, Q: QNetwork):
        self.Q = Q

    def act(self, obs: torch.Tensor) -> int:
        with eval_ctx(self.Q):
            q_values = self.Q(obs.unsqueeze(0)).squeeze(0)
            return int(torch.argmax(q_values).item())


class EpsGreedyAgent:
    def __init__(self, Q: QNetwork, eps: float):
        self.Q = Q
        self.greedy = GreedyAgent(self.Q)
        self.eps = eps

    def act(self, obs: torch.Tensor) -> int:
        if torch.rand(1) < self.eps:
            return int(torch.randint(self.Q.num_actions, tuple()).item())
        else:
            return self.greedy.act(obs)


def dqn():
    def make_env(train: bool):
        env_name = "Seaquest-v4"
        env = gym.make(env_name, frameskip=1)
        frame_skip = 3 if "SpaceInvaders" in env_name else 4
        env = AtariPreprocessing(
            env=env,
            noop_max=0,
            frame_skip=frame_skip,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        env = FrameStack(env, num_stack=4)
        env = TransformObservation(env, np.asarray)
        if train:
            env = TransformReward(env, np.sign)
        return env

    train_env, val_env = make_env(train=True), make_env(train=False)
    env_seed = 42
    obs, info = train_env.reset(seed=env_seed)

    batch_size = 32
    train_frames = int(10e6)
    val_every = int(1e5)
    val_episodes = 32
    buffer_capacity = int(1e5)
    max_eps, min_eps = 1.0, 0.1
    val_eps = 0.05
    gamma = 0.99

    replay_buffer = ReplayBuffer(train_env, capacity=buffer_capacity)
    Q = QNetwork(train_env)
    Q_optim = torch.optim.RMSprop(Q.parameters())

    train_agent = EpsGreedyAgent(Q, eps=max_eps)
    val_agent = EpsGreedyAgent(Q, eps=val_eps)

    device = torch.device("cpu")
    idx_seq = torch.arange(batch_size).to(device)

    def to_tensor(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    step = 0

    def should_stop():
        return step >= train_frames

    def train_step():
        nonlocal obs

        t = step / train_frames
        train_agent.eps = max_eps * (1.0 - t) + min_eps * t
        action = train_agent.act(to_tensor(obs))
        next_obs, reward, term, trunc, info = train_env.step(action)
        done = term or trunc

        tr = Transition(obs, action, reward, next_obs, done)
        replay_buffer.add(tr)

        obs = next_obs
        if done:
            obs, info = train_env.reset()

        batch = replay_buffer.sample(batch_size).to(device)

        Q_optim.zero_grad()

        value_pred = Q(batch.obs)[idx_seq, batch.act]
        with torch.no_grad():
            next_reward = torch.argmax(Q(batch.next_obs), dim=1)
        target = batch.reward + (1.0 - batch.done) * gamma * next_reward
        Q_loss = F.mse_loss(value_pred, target)

        Q_loss.backward()
        Q_optim.step()

    def val_step():
        ep_returns = []
        for ep in range(val_episodes):
            obs, info = val_env.reset()
            ep_return = 0
            while True:
                action = val_agent.act(to_tensor(obs))
                next_obs, reward, term, trunc, info = val_env.step(action)
                ep_return += reward
                obs = next_obs

                done = term or trunc
                if done:
                    break

            ep_returns.append(ep_return)

        ep_returns = np.array(ep_returns)
        print(
            f"[val] step={step}, mean={ep_returns.mean():.2f}, std={ep_returns.std():.2f}, min={ep_returns.min():.2f}, max={ep_returns.max():.2f}"
        )

    while True:
        # if step % val_every == 0:
        #     val_step()
        if should_stop():
            break
        train_step()
        step += 1
