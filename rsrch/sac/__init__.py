from typing import Callable, Iterable, Protocol

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as nn_F
from torch import Tensor, nn
from tqdm.auto import tqdm

from rsrch.rl import agent, gym
from rsrch.rl.data.buffer import StepBuffer
from rsrch.rl.data.rollout import EpisodeRollout, StepRollout
from rsrch.rl.data.step import StepBatch
from rsrch.rl.utils import fix_log_prob_
from rsrch.rl.utils.polyak import Polyak
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
    def __call__(self, obs: Tensor) -> D.Distribution:
        return super().__call__(obs)


class QNet(nn.Module):
    def __call__(self, obs: Tensor, act: Tensor) -> Tensor:
        return super().__call__(obs, act)


class VNet(nn.Module):
    def __call__(self, obs: Tensor) -> Tensor:
        return super().__call__(obs)

    def clone(self):
        ...


class MinQ(QNet):
    def __init__(self, Qs: Iterable[QNet]):
        super().__init__()
        self.Qs = nn.ModuleList(Qs)

    def __call__(self, obs: Tensor, act: Tensor) -> Tensor:
        qs = [Q(obs, act) for Q in self.Qs]
        return torch.min(torch.stack(qs), dim=0).values


class Agent(nn.Module, agent.Agent):
    pi: Policy
    V: VNet
    Q: MinQ

    def reset(self):
        ...

    def act(self, obs: Tensor) -> Tensor:
        with eval_ctx(self):
            return self.pi(obs.unsqueeze(0))[0]


class Config:
    env_name: str
    seed: int
    alpha: float


class Trainer:
    def __init__(self, conf: Config):
        self.replay_buf_size = int(1e5)
        self.batch_size = 128
        self.val_every_steps = int(10e3)
        self.env_iters_per_step = 1
        self.prefill = int(1e3)
        self.gamma = 0.99
        self.device = torch.device("cuda:0")
        self.tau = 0.995
        self.val_episodes = 16
        self.alpha = conf.alpha

    def train(self, sac: Agent, sac_data: Data):
        self.agent, self.data = sac, sac_data

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
        self.train_env = gym.wrappers.CollectSteps(self.train_env, self.replay_buf)
        self.env_iter = iter(StepRollout(self.train_env, self.agent))

        prefill_agent = agent.RandomAgent(self.train_env)
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
        self.pi, self.V, self.Q = self.agent.pi, self.agent.V, self.agent.Q

        self.pi = self.pi.to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=1e-3)

        self.V = self.V.to(self.device)
        self.V_optim = torch.optim.Adam(self.V.parameters(), lr=1e-3)
        self.target_V: VNet = self.V.clone()
        self.V_polyak = Polyak(self.V, self.target_V, self.tau)

        self.Q = self.Q.to(self.device)
        self.Q_opt_pairs = []
        for Qi in self.Q.Qs:
            Qi_optim = torch.optim.Adam(Qi.parameters(), lr=1e-3)
            self.Q_opt_pairs.append((Qi, Qi_optim))

    def init_extras(self):
        self.exp_dir = ExpDir()
        self.board = Board(root_dir=self.exp_dir / "board")
        self.pbar = tqdm(desc="SAC")

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
                cur_env = gym.wrappers.RenderCollection(cur_env)

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

        # Value net loss
        self.V.train()
        v_preds = self.V(batch.obs)
        cur_policy = self.pi(batch.obs)
        with eval_ctx(self.Q):
            cur_act = cur_policy.sample()
            logp = cur_policy.log_prob(cur_act)
            v_targets = self.Q(batch.obs, cur_act) - self.alpha * logp
        V_loss = nn_F.mse_loss(v_preds, v_targets)

        self.V_optim.zero_grad(set_to_none=True)
        V_loss.backward()
        self.V_optim.step()

        self.board.add_scalar("train/V_loss", V_loss, step=self.step_idx)

        # Q net loss
        with eval_ctx(self.target_V):
            gamma_t = self.gamma * (1.0 - batch.term.float())
            q_targets = batch.reward + gamma_t * self.target_V(batch.next_obs)

        for i, (Qi, Qi_optim) in enumerate(self.Q_opt_pairs):
            Qi.train()
            q_preds = Qi(batch.obs, batch.act)
            Qi_loss = nn_F.mse_loss(q_preds, q_targets)

            Qi_optim.zero_grad(set_to_none=True)
            Qi_loss.backward()
            Qi_optim.step()

            self.board.add_scalar(f"train/Q{i}_loss", Qi_loss, step=self.step_idx)

        # Policy net loss
        self.pi.train()
        cur_act_r = cur_policy.rsample()
        pi_logits = cur_policy.log_prob(cur_act_r)

        with eval_ctx(self.Q, no_grad=False):
            pi_target = self.Q(batch.obs, cur_act_r)
        pi_loss = (pi_logits - pi_target).mean()

        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_optim.step()

        self.board.add_scalar("train/pi_loss", pi_loss, step=self.step_idx)

        # Value net - Polyak step
        self.V_polyak.step()