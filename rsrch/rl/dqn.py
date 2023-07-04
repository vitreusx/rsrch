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
import copy
from tqdm.auto import tqdm
from itertools import count


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
    def __init__(
        self, env: gym.Env, capacity: int, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Box)
        obs_type = env.observation_space.dtype
        self.obs = np.empty((capacity, *env.observation_space.shape), dtype=obs_type)

        assert isinstance(env.action_space, gym.spaces.Discrete)
        act_type = env.action_space.dtype
        self.act = np.empty((capacity,), dtype=act_type)

        self.reward = np.empty((capacity,), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.done = np.empty((capacity,), dtype=np.float32)

        self.obs = torch.from_numpy(self.obs).to(device=device)
        self.act = torch.from_numpy(self.act).to(device=device)
        self.reward = torch.from_numpy(self.reward).to(device=device)
        self.next_obs = torch.from_numpy(self.next_obs).to(device=device)
        self.done = torch.from_numpy(self.done).to(device=device)

        self.capacity = capacity
        self.size = 0
        self.ins_idx = 0

    def __len__(self):
        return self.size

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


class ProprioQNet(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        assert isinstance(env.observation_space, gym.spaces.Box)
        in_features = int(np.prod(env.observation_space.shape))

        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.num_actions = int(env.action_space.n)

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class AtariQNet(nn.Module):
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
    def __init__(self, Q: nn.Module):
        self.Q = Q

    def act(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.Q(obs.unsqueeze(0)).squeeze(0)
            return torch.argmax(q_values).item()


class EpsGreedyAgent:
    def __init__(self, Q: nn.Module, eps: float):
        self.Q = Q
        self.greedy = GreedyAgent(self.Q)
        self.eps = eps

    def act(self, obs: torch.Tensor) -> int:
        if torch.rand(1) < self.eps:
            return torch.randint(self.Q.num_actions, tuple()).item()
        else:
            return self.greedy.act(obs)


def polyak_update(target: nn.Module, source: nn.Module, tau: float):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        new_val = tau * target_p.data + (1.0 - tau) * source_p.data
        target_p.data.copy_(new_val)


def dqn():
    def make_atari_env(name: str, train: bool):
        env = gym.make(name, frameskip=1)
        frame_skip = 3 if "SpaceInvaders" in name else 4
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

    def make_env(train: bool):
        return gym.make("CartPole-v1")
        # return make_atari_env("Pong-v4", train)

    train_env, val_env = make_env(train=True), make_env(train=False)
    # env_seed = 42
    obs, info = train_env.reset()

    batch_size = 128
    train_frames = int(1e6)
    train_episodes = int(5e3)
    val_every = int(1e3)
    val_episodes = 32
    buffer_capacity = int(1e4)
    max_eps, min_eps = 0.9, 0.05
    eps_decay = 1e-3
    val_eps = 0.05
    gamma = 0.99
    tau = 0.995

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    replay_buffer = ReplayBuffer(train_env, capacity=buffer_capacity, device=device)
    # QNet = AtariQNet
    QNet = ProprioQNet
    Q = QNet(train_env).to(device)
    target_Q = QNet(train_env).to(device)

    Q_optim = torch.optim.AdamW(Q.parameters(), lr=1e-4, amsgrad=True)
    train_agent = EpsGreedyAgent(Q, eps=max_eps)
    val_agent = EpsGreedyAgent(Q, eps=val_eps)

    def to_tensor(x, dtype=None):
        return torch.as_tensor(x, dtype=dtype, device=device)

    step, ep_idx = 0, 0

    def should_stop():
        return step >= train_frames or ep_idx >= train_episodes

    def train_step():
        nonlocal obs, ep_idx

        train_agent.eps = min_eps + (max_eps - min_eps) * np.exp(-eps_decay * step)
        action = train_agent.act(to_tensor(obs))
        next_obs, reward, term, trunc, info = train_env.step(action)
        done = term or trunc

        tr = Transition(obs, action, float(reward), next_obs, done)
        replay_buffer.add(tr)

        obs = next_obs
        if done:
            ep_idx += 1
            obs, info = train_env.reset()

        if len(replay_buffer) < batch_size:
            return

        batch = replay_buffer.sample(batch_size).to(device)

        value_pred = Q(batch.obs).gather(1, batch.act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_V = target_Q(batch.next_obs).max(dim=1)[0]
        target = batch.reward + (1.0 - batch.done) * gamma * next_V
        loss = F.smooth_l1_loss(value_pred, target)

        Q_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(Q.parameters(), 100.0)
        Q_optim.step()

        polyak_update(target_Q, Q, tau)

    def val_step():
        ep_returns = []
        for ep in range(val_episodes):
            obs, info = val_env.reset()
            ep_return = 0.0
            while True:
                action = val_agent.act(to_tensor(obs))
                next_obs, reward, term, trunc, info = val_env.step(action)
                ep_return = ep_return + float(reward)
                obs = next_obs

                done = term or trunc
                if done:
                    break

            ep_returns.append(ep_return)

        ep_returns = np.array(ep_returns)
        print(
            f"[val] step={step}, ep={ep_idx}, mean={ep_returns.mean():.2f}, std={ep_returns.std():.2f}, min={ep_returns.min():.2f}, max={ep_returns.max():.2f}"
        )

    while True:
        if step % val_every == 0:
            val_step()
        if should_stop():
            break
        train_step()
        step += 1
    val_step()
