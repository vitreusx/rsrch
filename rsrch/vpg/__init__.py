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

from rsrch.rl import agent, gym, wrappers
from rsrch.rl.data import EpisodeRollout, StepBatch, StepRollout
from rsrch.rl.data import transforms as T
from rsrch.rl.data.step import TensorStep
from rsrch.rl.spec import EnvSpec
from rsrch.utils import data
from rsrch.utils.board import Board
from rsrch.utils.eval_ctx import eval_ctx
from rsrch.utils.exp_dir import ExpDir


class Data(Protocol):
    def val_env(self, device=None) -> gym.Env:
        ...

    def train_env(self, device=None) -> gym.Env:
        ...


class Policy(nn.Module):
    __call__: Callable[[Tensor], D.Distribution]


class ValueNet(nn.Module):
    __call__: Callable[[Tensor], Tensor]


class Agent(nn.Module):
    pi: Policy
    V: ValueNet

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self):
            act_dist = self.pi(obs.unsqueeze(0))
            return act_dist.sample()[0]


def rev_cumsum(A: np.ndarray, gamma: float) -> np.ndarray:
    """Given :math:`A = [A_1, ..., A_n]`, return :math:`S = [S_1, ..., S_n]`, where :math:`S_i = A_i + \gamma A_{i+1} + \\cdots + \gamma^{n-i} A_n`."""
    return scipy.signal.lfilter([1], [1, -gamma], A[::-1])[::-1]


class VPGTrainer:
    def __init__(self):
        self.env_steps_per_update = int(2**10)
        self.batch_size = 128
        assert self.env_steps_per_update % self.batch_size == 0
        self.device = torch.device("cuda")
        self.val_every_epoch = 16
        self.val_episodes = 32
        self.gamma = 0.95
        self.td_lambda = 0.97

    def train(self, vpg: Agent, vpg_data: Data):
        self.vpg, self.vpg_data = vpg, vpg_data

        self.init_envs()
        self.init_data()
        self.init_model()
        self.init_extras()
        self.loop()

    def init_envs(self):
        self.val_env = self.vpg_data.val_env(self.device)
        self.train_env = self.vpg_data.train_env(self.device)

    def init_data(self):
        self.ep_iter = None

    def init_model(self):
        self.vpg = self.vpg.to(self.device)
        self.policy_optim = torch.optim.Adam(self.vpg.pi.parameters(), lr=1e-3)
        self.value_optim = torch.optim.Adam(self.vpg.V.parameters(), lr=1e-3)

    def init_extras(self):
        self.exp_dir = ExpDir()
        self.board = Board(root_dir=self.exp_dir / "board")
        self.pbar = tqdm(desc="VPG")

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

            val_ep_rollout = EpisodeRollout(cur_env, self.vpg, num_episodes=1)
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
        # First, we perform environment interaction.
        # For each episode:
        #   \tau = [s_0, a_0, r_1, s_1, ..., s_{n-1}, a_{n-1}, r_n, s_n]
        # we collect:
        # - rewards [r_1, ..., r_n];
        # - value estimates [V(s_0), ..., V(s_{n-1})];
        # - whether s_n is terminal state or not;
        # - log-probabilities [log \pi(a_0 | s_0), ...];
        # - length n;
        # - final value V(s_n), either 0 if s_n is terminal or the estimate from the neural network if truncated.
        # We batch computation of V(s_i) and log \pi(a_i | s_i), possibly across episodes. This allows us to evaluate NNs with proper batch sizes, instead of, say, doing it for each step or each episode or all episodes.
        batch, total_steps = [], 0
        rews, term, v, logp, lengths, final_v = [], [], [], [], [], []
        while True:
            # Run for precisely env_steps_per_update, possibly truncating the episode.
            if total_steps >= self.env_steps_per_update:
                break

            # If starting a new episode, initialize env iterator and per-episode data.
            if self.ep_iter is None:
                self._steps = StepRollout(self.train_env, self.vpg, num_episodes=1)
                self._steps = self._steps.map(T.ToTensorStep())
                self.step_iter = iter(self._steps)
                rews.append([])
                term.append(False)
                lengths.append(0)

            # Iterate over the episode
            step: TensorStep
            for step in self.step_iter:
                batch.append(step)
                # If we've accumulated enough observations, compute value estimates and log-probs for this batch and clear it.
                if len(batch) >= self.batch_size:
                    batch = StepBatch.collate(batch).to(self.device)
                    batch_v = self.vpg.V(batch.obs)
                    v.append(batch_v)
                    batch_logp = self.vpg.pi(batch.obs).log_prob(batch.act)
                    logp.append(batch_logp)
                    batch = []

                rews[-1].append(step.reward)
                term[-1] |= step.term
                lengths[-1] += 1
                total_steps += 1

                # Run for precisely env_steps_per_update, possibly truncating the episode.
                if total_steps >= self.env_steps_per_update:
                    break

            # Convert reward list to np.ndarray
            rews[-1] = np.array(rews[-1])

            # Compute final V(s_n)
            if step.term:
                ep_final_v = 0.0
            else:
                with eval_ctx(self.vpg.V):
                    ep_final_v = self.vpg.V(step.next_obs.unsqueeze(0)).item()
            final_v.append(ep_final_v)

            # Reset the episode iterator.
            self.ep_iter = None

        # Concatenate the batches for v and logp into a single array.
        v = torch.cat(v)
        logp = torch.cat(logp)

        # Now, for each episode, compute GAE advantages and returns for the loss functions.
        ep_vs = v.split(lengths)
        ret, adv = [], []
        for ep_rews, ep_v, ep_final_v in zip(rews, ep_vs, final_v):
            vals = ep_v.detach().cpu().numpy()
            vals = np.append(vals, ep_final_v)
            deltas = (ep_rews + self.gamma * vals[1:]) - vals[:-1]
            ep_adv = rev_cumsum(deltas, self.gamma * self.td_lambda)
            adv.append(torch.as_tensor(ep_adv.copy()))

            ep_rews_ = np.append(ep_rews, ep_final_v)
            ep_ret = rev_cumsum(ep_rews_, self.gamma)[:-1]
            ret.append(torch.as_tensor(ep_ret.copy()))

        ret, adv = torch.cat(ret), torch.cat(adv)

        # Compute the losses
        policy_loss = -(adv.type_as(logp) * logp).mean()
        value_loss = nn_F.mse_loss(v, ret.type_as(v))

        # Optimize policy network
        self.policy_optim.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optim.step()

        # Optimize value network
        self.value_optim.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optim.step()

        self.board.add_scalar("train/policy_loss", policy_loss, step=self.epoch_idx)
        self.board.add_scalar("train/value_loss", value_loss, step=self.epoch_idx)
