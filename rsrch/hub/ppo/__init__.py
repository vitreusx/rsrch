import argparse
from dataclasses import dataclass
from itertools import count
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import scipy.signal
import torch
import torch.distributions as D
import torch.nn.functional as nn_F
from torch import Tensor, autograd, nn
from tqdm.auto import tqdm

from rsrch.rl import gym, wrappers
from rsrch.rl.data import EpisodeRollout, StepBatch, StepRollout
from rsrch.rl.data import transforms as T
from rsrch.rl.data.step import TensorStep
from rsrch.rl.gym import agents
from rsrch.rl.gym.spec import EnvSpec
from rsrch.utils import data
from rsrch.utils.board import Board
from rsrch.utils.eval_ctx import eval_ctx
from rsrch.utils.exp_dir import ExpDir
from rsrch.vpg import rev_cumsum


class Data(Protocol):
    def val_env(self, device=None) -> gym.Env:
        ...

    def train_env(self, device=None) -> gym.Env:
        ...


class Policy(nn.Module):
    def __call__(self, obs: Tensor) -> D.Distribution:
        return super().__call__(obs)


class ValueNet(nn.Module):
    def __call__(self, obs: Tensor) -> Tensor:
        return super().__call__(obs)


class Agent(nn.Module, agents.Agent):
    pi: Policy
    V: ValueNet

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self):
            act_dist = self.pi(obs.unsqueeze(0))
            return act_dist.sample()[0]


class PPOTrainer:
    def __init__(self):
        self.env_steps_per_update = int(2**10)
        self.batch_size = 128
        assert self.env_steps_per_update % self.batch_size == 0
        self.device = torch.device("cuda")
        self.val_every_epoch = 64
        self.val_episodes = 16
        self.gamma = 0.99
        self.td_lambda = 0.97
        self.clip_eps = 0.2
        self.pi_optim_iters = 40
        self.v_optim_iters = 40

    def train(self, ppo: Agent, ppo_data: Data):
        self.ppo, self.ppo_data = ppo, ppo_data

        self.init_envs()
        self.init_data()
        self.init_model()
        self.init_extras()
        self.loop()

    def init_envs(self):
        self.val_env = self.ppo_data.val_env(self.device)
        self.train_env = self.ppo_data.train_env(self.device)

    def init_data(self):
        self.ep_iter = None

    def init_model(self):
        self.ppo = self.ppo.to(self.device)
        self.pi = self.ppo.pi
        self.pi_optim = torch.optim.Adam(self.ppo.pi.parameters(), lr=1e-3)
        self.V_optim = torch.optim.Adam(self.ppo.V.parameters(), lr=1e-3)

    def init_extras(self):
        self.exp_dir = ExpDir()
        self.board = Board(root_dir=self.exp_dir / "board")
        self.pbar = tqdm(desc="PPO")

    def loop(self):
        for self.epoch_idx in count():
            if self.epoch_idx % self.val_every_epoch == 0:
                self.val_epoch()
            self.train_epoch()
            self.pbar.update()

    def val_epoch(self):
        returns = []
        for ep_idx in range(self.val_episodes):
            cur_env = self.val_env
            if ep_idx == 0:
                cur_env = wrappers.RenderCollection(cur_env)

            val_ep_rollout = EpisodeRollout(cur_env, self.ppo, num_episodes=1)
            val_ep = next(iter(val_ep_rollout))
            ep_R = sum(val_ep.reward)
            returns.append(ep_R)

            if ep_idx == 0:
                video = cur_env.frame_list
                video_fps = cur_env.metadata.get("render_fps", 30.0)
                self.board.add_video(
                    "val/video", video, step=self.epoch_idx, fps=video_fps
                )

        self.board.add_scalar("val/returns", np.mean(returns), step=self.epoch_idx)

    def train_epoch(self):
        step_idx = 0
        obs, act, rew, term, lengths, final_v = [], [], [], [], [], []

        # Do environment interaction
        while True:
            # Run for precisely env_steps_per_update, possibly truncating the episode.
            if step_idx >= self.env_steps_per_update:
                break

            # If starting a new episode, initialize env iterator and per-episode data.
            if self.ep_iter is None:
                self._steps = StepRollout(self.train_env, self.ppo, num_episodes=1)
                self._steps = self._steps.map(T.ToTensorStep())
                self.step_iter = iter(self._steps)
                lengths.append(0)
                rew.append([])

            # Iterate over the episode
            step: TensorStep
            for step in self.step_iter:
                obs.append(step.obs)
                act.append(step.act)
                rew[-1].append(step.reward)
                lengths[-1] += 1

                step_idx += 1
                if step_idx >= self.env_steps_per_update:
                    break

            # Register whether episode was truncated or terminated
            term.append(step.term)
            rew[-1] = np.array(rew[-1])

            # Compute final V(s_n)
            if step.term:
                ep_final_v = 0.0
            else:
                with eval_ctx(self.ppo.V):
                    ep_final_v = self.ppo.V(step.next_obs.unsqueeze(0)).item()
            final_v.append(ep_final_v)

            # Reset the episode iterator.
            self.ep_iter = None

        obs, act = torch.stack(obs), torch.stack(act)

        # Compute value estimates and log-probabilities of selected actions
        def compute_v():
            v = []
            for off in range(0, len(obs), self.batch_size):
                idxes = slice(off, off + self.batch_size)
                batch_v = self.ppo.V(obs[idxes])
                v.append(batch_v)
            return torch.cat(v)

        def compute_logp():
            logp = []
            for off in range(0, len(obs), self.batch_size):
                idxes = slice(off, off + self.batch_size)
                batch_logp = self.pi(obs[idxes]).log_prob(act[idxes])
                logp.append(batch_logp)
            return torch.cat(logp)

        with torch.no_grad():
            v = compute_v()
            logp = compute_logp()

        # Now, for each episode, compute advantages and returns.
        ep_vs = v.split(lengths)

        ret, adv = [], []
        for ep_rew, ep_v, ep_final_v in zip(rew, ep_vs, final_v):
            vals = ep_v.detach().cpu().numpy()
            vals = np.append(vals, ep_final_v)
            deltas = (ep_rew + self.gamma * vals[1:]) - vals[:-1]
            ep_adv = rev_cumsum(deltas, self.gamma * self.td_lambda)
            adv.append(torch.as_tensor(ep_adv.copy()))

            ep_rews_ = np.append(ep_rew, ep_final_v)
            ep_ret = rev_cumsum(ep_rews_, self.gamma)[:-1]
            ret.append(torch.as_tensor(ep_ret.copy()))

        ret = torch.cat(ret).type_as(v)
        adv = torch.cat(adv).type_as(logp)

        # Optimize the policy
        for _ in range(self.pi_optim_iters):
            ratios = []
            for off in range(0, len(obs), self.batch_size):
                idxes = slice(off, off + self.batch_size)
                batch_logp = self.pi(obs[idxes]).log_prob(act[idxes])
                batch_ratio = torch.exp(batch_logp - logp[idxes])
                ratios.append(batch_ratio)
            r = ratios = torch.cat(ratios)

            clip_r = torch.clamp(r, 1 - self.clip_eps, 1 + self.clip_eps)
            clip_loss = -torch.min(r * adv, clip_r * adv).mean()

            self.pi_optim.zero_grad(set_to_none=True)
            clip_loss.backward()
            self.pi_optim.step()

        # Optimize value network
        for _ in range(self.v_optim_iters):
            self.V_optim.zero_grad(set_to_none=True)
            v_loss = nn_F.mse_loss(compute_v(), ret)
            v_loss.backward()
            self.V_optim.step()

        self.board.add_scalar("train/clip_loss", clip_loss, step=self.epoch_idx)
        self.board.add_scalar("train/V_loss", v_loss, step=self.epoch_idx)
