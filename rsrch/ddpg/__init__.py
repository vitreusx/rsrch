from typing import Callable, Protocol

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as nn_F
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch.rl import agent, gym, wrappers
from rsrch.rl.data.buffer import StepBuffer
from rsrch.rl.data.rollout import EpisodeRollout, StepRollout
from rsrch.rl.data.step import StepBatch
from rsrch.rl.utils.polyak import Polyak
from rsrch.rl.wrappers import CollectSteps
from rsrch.utils import data
from rsrch.utils.board import Board
from rsrch.utils.eval_ctx import eval_ctx
from rsrch.utils.exp_dir import ExpDir


class Data(Protocol):
    def val_env(self, device: torch.device) -> gym.Env:
        ...

    def train_env(self, device: torch.device) -> gym.Env:
        ...


class Policy(nn.Module):
    def __call__(self, obs: Tensor) -> Tensor:
        return super().__call__(obs)

    def clone(self):
        ...


class QNet(nn.Module):
    def __call__(self, obs: Tensor, act: Tensor) -> Tensor:
        return super().__call__(obs, act)

    def clone(self):
        ...


class Agent(nn.Module, agent.Agent):
    pi: Policy
    Q: QNet

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self):
            return self.pi(obs.unsqueeze(0))[0]


class Trainer:
    def __init__(self):
        self.replay_buf_size = int(1e5)
        self.batch_size = 128
        self.val_every_steps = int(10e3)
        self.env_iters_per_step = 1
        self.prefill = int(1e3)
        self.gamma = 0.99
        self.device = torch.device("cuda")
        self.tau = 0.995
        self.noise_std = 0.1
        self.val_episodes = 16

    def train(self, ddpg: Agent, ddpg_data: Data):
        self.agent, self.data = ddpg, ddpg_data

        self.init_envs()
        self.init_data()
        self.init_model()
        self.init_extras()

        self.loop()

    def init_envs(self):
        self.val_env = self.data.val_env(self.device)
        self.train_env = self.data.train_env(self.device)

    def init_data(self):
        self.replay_buf = StepBuffer(self.train_env, self.replay_buf_size)
        self.train_env = CollectSteps(self.train_env, self.replay_buf)

        act_shape = self.train_env.action_space.shape
        noise_mean = torch.zeros(*act_shape).to(self.device)
        noise_std = self.noise_std * torch.ones(*act_shape).to(self.device)
        noise_fn = lambda: D.Normal(noise_mean, noise_std)
        train_agent = agent.WithNoise(self.agent, noise_fn)
        self.env_iter = iter(StepRollout(self.train_env, train_agent))

        prefill_agent = agent.RandomAgent(self.train_env)
        prefill_agent = agent.ToTensor(prefill_agent, self.device)
        prefill_iter = iter(StepRollout(self.train_env, prefill_agent))
        while len(self.replay_buf) < self.prefill:
            _ = next(prefill_iter)

        self.loader = data.DataLoader(
            dataset=self.replay_buf,
            batch_size=self.batch_size,
            sampler=data.RandomInfiniteSampler(self.replay_buf),
            collate_fn=StepBatch.collate,
        )
        self.batch_iter = iter(self.loader)

    def init_model(self):
        self.pi, self.Q = self.agent.pi, self.agent.Q

        self.pi = self.pi.to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=1e-3)
        self.target_pi: Policy = self.pi.clone()
        self.pi_polyak = Polyak(self.pi, self.target_pi, self.tau)

        self.Q = self.Q.to(self.device)
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.target_Q: QNet = self.Q.clone()
        self.Q_polyak = Polyak(self.Q, self.target_Q, self.tau)

    def init_extras(self):
        self.exp_dir = ExpDir()
        self.board = Board(root_dir=self.exp_dir / "board")
        self.pbar = tqdm(desc="DDPG")

    def loop(self):
        self.step_idx = 0
        while True:
            if self.step_idx % self.val_every_steps == 0:
                self.val_epoch()
            self.train_step()
            self.step_idx += 1
            self.pbar.update()

    def val_epoch(self):
        val_ep_returns = []
        for ep_idx in range(self.val_episodes):
            cur_env = self.val_env
            if ep_idx == 0:
                cur_env = wrappers.RenderCollection(cur_env)

            ep_source = EpisodeRollout(cur_env, self.agent, num_episodes=1)
            episode = next(iter(ep_source))
            ep_R = sum(episode.reward)
            val_ep_returns.append(ep_R)

            if ep_idx == 0:
                video = cur_env.frame_list
                video_fps = cur_env.metadata.get("render_fps", 30.0)
                self.board.add_video(
                    "val/video", video, step=self.step_idx, fps=video_fps
                )

        self.board.add_scalar(
            "val/returns", np.mean(val_ep_returns), step=self.step_idx
        )

    def train_step(self):
        for _ in range(self.env_iters_per_step):
            _ = next(self.env_iter)

        batch: StepBatch = next(self.batch_iter).to(self.device)

        preds = self.Q(batch.obs, batch.act)
        with torch.no_grad():
            next_act = self.target_pi(batch.next_obs)
            next_preds = self.target_Q(batch.next_obs, next_act)
            gamma = self.gamma * (1.0 - batch.term.float())
            targets = batch.reward + gamma * next_preds
        critic_loss = nn_F.mse_loss(preds, targets)

        self.Q_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.Q_optim.step()
        self.Q_polyak.step()

        self.board.add_scalar("train/critic_loss", critic_loss, step=self.step_idx)

        actor_loss = -self.Q(batch.obs, self.pi(batch.obs)).mean()

        self.pi_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.pi_optim.step()
        self.pi_polyak.step()

        self.board.add_scalar("train/actor_loss", actor_loss, step=self.step_idx)
