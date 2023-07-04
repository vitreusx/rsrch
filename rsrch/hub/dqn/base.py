import argparse
import datetime
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn
from tqdm.auto import tqdm, trange

import rsrch.rl.agents as agents
import rsrch.rl.gym as gym
import rsrch.utils.data as data
from rsrch.rl import wrappers
from rsrch.rl.data import EpisodeRollout, StepBatch, StepBuffer, StepRollout
from rsrch.rl.spec import EnvSpec
from rsrch.rl.utils.polyak import Polyak
from rsrch.utils.board import Board

from .agent import QAgent
from .loss import DQNLoss
from .types import QNetwork


class BaseQ(nn.Module, QNetwork):
    def __init__(self, spec: EnvSpec):
        super().__init__()

        assert isinstance(spec.action_space, gym.spaces.Discrete)
        self.num_actions = int(spec.action_space.n)

        in_features = int(np.prod(spec.observation_space.shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class DQN(nn.Module):
    def __init__(self, spec: EnvSpec):
        super().__init__()
        self.Q, self.target_Q = BaseQ(spec), BaseQ(spec)
        self.target_Q.load_state_dict(self.Q.state_dict())


class DQNData:
    def __init__(self, name: str, seed=42):
        self.name = name
        self.seed = seed
        self.spec = self.val_env()

    def train_env(self, device=None) -> gym.Env:
        return self.val_env(device=device)

    def val_env(self, device=None) -> gym.Env:
        env = gym.make(self.name, render_mode="rgb_array")
        env = wrappers.ToTensor(env, device)
        env.reset(seed=self.seed)
        return env


class DQNTrainer:
    def __init__(self):
        self.batch_size = 128
        self.train_steps = int(1e6)
        self.train_episodes = int(5e3)
        self.val_every_steps = int(10e3)
        self.val_episodes = 32
        self.buffer_capacity = int(1e4)
        self.max_eps, self.min_eps = 0.9, 0.05
        self.eps_step_decay = 1e-3
        self.val_eps = 0.05
        self.gamma = 0.99
        self.tau = 0.995
        self.clip_grad = 100.0
        self.prefill = int(1e3)
        self.device = torch.device("cuda")
        self.exp_root = None
        self.precision = "fp32"
        self.pbar_enabled = True

    def train(self, dqn: DQN, dqn_data: DQNData):
        dqn = dqn.to(self.device)

        val_env = dqn_data.val_env(self.device)
        train_env = dqn_data.train_env(self.device)

        rand_agent = agents.RandomAgent(val_env)
        rand_agent = agents.ToTensor(rand_agent, self.device)
        train_agent = agents.EpsAgent(QAgent(dqn.Q), rand_agent, self.max_eps)
        val_agent = agents.EpsAgent(QAgent(dqn.Q), rand_agent, self.val_eps)

        replay_buffer = StepBuffer(train_env, self.buffer_capacity)
        train_env = wrappers.CollectSteps(train_env, replay_buffer)
        env_interaction = iter(StepRollout(train_env, train_agent))

        train_loader = data.DataLoader(
            dataset=replay_buffer,
            sampler=data.BatchSampler(
                sampler=data.RandomInfiniteSampler(replay_buffer),
                batch_size=self.batch_size,
                drop_last=False,
            ),
            batch_size=None,
        )
        batch_iter = iter(train_loader)

        loss = DQNLoss(dqn.Q, self.gamma, dqn.target_Q)
        optim = torch.optim.AdamW(dqn.Q.parameters(), lr=1e-4, amsgrad=True)
        amp_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        amp_enabled = amp_dtype != torch.float32
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        eps_sched = agents.EpsScheduler(
            train_agent, self.max_eps, self.min_eps, self.eps_step_decay
        )
        polyak = Polyak(dqn.Q, dqn.target_Q, self.tau)

        if self.exp_root is None:
            now = datetime.datetime.now()
            host = socket.gethostname()
            self.exp_root = Path(f"runs/{now:%b%d_%H-%M-%S}_{host}")

        board = Board(root_dir=(self.exp_root / "board"))

        pbar = tqdm(desc="DQN", position=0, disable=not self.pbar_enabled)

        step_idx = 0

        prefill_iter = iter(StepRollout(train_env, rand_agent))
        for _ in range(self.prefill):
            _ = next(prefill_iter)

        def train_step():
            nonlocal step_idx

            _ = next(env_interaction)

            batch: StepBatch = next(batch_iter)
            batch = batch.to(self.device)

            with torch.autocast(
                device_type=self.device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                batch_loss = loss(batch)

            optim.zero_grad(set_to_none=True)
            scaler.scale(batch_loss).backward()
            if self.clip_grad is not None:
                nn.utils.clip_grad.clip_grad_value_(dqn.Q.parameters(), self.clip_grad)
            scaler.step(optim)

            polyak.step()
            eps_sched.step()
            scaler.update()

            board.add_scalar("train/loss", batch_loss, step=step_idx)
            board.add_scalar("train/eps", eps_sched.cur_eps, step=step_idx)
            board.add_scalar("train/buffer", len(replay_buffer), step=step_idx)

            step_idx += 1
            pbar.update()

        def val_epoch():
            val_ep_returns = []
            val_pbar = tqdm(
                total=self.val_episodes,
                position=1,
                leave=False,
                desc="Val",
                disable=not self.pbar_enabled,
            )

            for ep_idx in range(self.val_episodes):
                cur_env = val_env
                if ep_idx == 0:
                    cur_env = wrappers.RenderCollection(cur_env)

                val_ep_iter = EpisodeRollout(cur_env, val_agent, num_episodes=1)
                val_ep = next(iter(val_ep_iter))
                ep_R = sum(val_ep.reward)
                val_ep_returns.append(ep_R)

                if ep_idx == 0:
                    video = cur_env.frame_list
                    video_fps = cur_env.metadata.get("render_fps", 30.0)
                    board.add_video("val/video", video, step=step_idx, fps=video_fps)

                val_pbar.update()

            val_ep_returns = np.array(val_ep_returns)
            board.add_scalar("val/returns", val_ep_returns.mean(), step=step_idx)

        while True:
            if step_idx % self.val_every_steps == 0:
                val_epoch()

            if step_idx >= self.train_steps:
                break

            train_step()


@dataclass
class Config:
    env_name: str
    seed: int

    @staticmethod
    def from_argv():
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", type=str, default="LunarLander-v2")
        parser.add_argument("-seed", type=int, default=42)

        args = parser.parse_args()

        return Config(
            env_name=args.env_name,
            seed=args.seed,
        )


def main(conf: Optional[Config] = None):
    if conf is None:
        conf = Config.from_argv()

    dqn_data = DQNData(name=conf.env_name, seed=conf.seed)
    dqn = DQN(spec=dqn_data.spec)
    trainer = DQNTrainer()
    trainer.train(dqn, dqn_data)


if __name__ == "__main__":
    main()
