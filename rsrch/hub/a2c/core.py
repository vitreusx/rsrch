from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

import rsrch.distributions as D
import rsrch.rl.data.transforms as T
from rsrch.exp.board import Board
from rsrch.rl import agents, gym
from rsrch.rl.api import Agent
from rsrch.rl.data import ListSeq, TensorSeq, interact
from rsrch.rl.utils.polyak import Polyak
from rsrch.utils.cron import Every
from rsrch.utils.stats import Stats


class ActorAgent(Agent):
    def __init__(self, policy_fn):
        super().__init__()
        self.policy_fn = policy_fn

    def reset(self):
        pass

    def observe(self, obs):
        self._cur_obs = obs

    def policy(self):
        with torch.inference_mode():
            batch = self._cur_obs.unsqueeze(0)
            return self.policy_fn(batch).sample().squeeze(0)


class Actor:
    def __call__(self, obs: Tensor) -> D.Distribution:
        ...


class Critic(nn.Module):
    def __call__(self, obs: Tensor) -> Tensor:
        ...


class A2C(ABC):
    def train(self):
        self.setup()
        self.train_loop()

    def setup(self):
        self.setup_vars()
        self.setup_data()
        self.setup_models()
        self._post_setup_models()
        self.setup_agents()
        self.setup_extras()

    @abstractmethod
    def setup_vars(self):
        self.val_every: int = ...
        self.train_steps: int = ...
        self.val_episodes: int = ...
        self.gamma: float = ...
        self.log_every: int = ...
        self.batch_size: int = ...
        self.device: torch.device = ...
        self.critic_steps: int = ...
        self.copy_critic_every: int = ...
        self.gae_lambda: float = ...

    @abstractmethod
    def setup_data(self):
        self.train_env: gym.Env = ...
        self.val_env: gym.Env = ...

    @abstractmethod
    def setup_models(self):
        self.actor: Actor = ...
        self.actor_opt: Optimizer = ...
        self.critic: Critic = ...
        self.target_critic: Critic = ...
        self.critic_opt: Optimizer = ...

    def _post_setup_models(self):
        self.critic_polyak = Polyak(
            source=self.critic,
            target=self.target_critic,
            every=self.copy_critic_every,
        )

    def setup_agents(self):
        self.train_agent = ActorAgent(self.actor)
        self.val_agent = self.train_agent
        self.env_iter = iter(interact.steps_ex(self.train_env, self.train_agent))

    @abstractmethod
    def setup_extras(self):
        self.board: Board = ...

    def train_loop(self):
        self.step = 0
        self._val_epoch = Every(lambda: self.step, self.val_every, self.val_epoch)
        self.should_log = Every(lambda: self.step, self.log_every)
        self.pbar = tqdm(desc="A2C", total=self.train_steps)

        while True:
            self._val_epoch()
            if self.is_finished:
                break
            self.train_step()

            self.step += 1
            self.pbar.update()

    @property
    def is_finished(self):
        return self.step >= self.train_steps

    def val_epoch(self):
        val_rets = []
        for ep_idx in range(self.val_episodes):
            val_ep = interact.one_episode(self.val_env, self.val_agent)
            val_ret = sum(val_ep.reward)
            val_rets.append(val_ret)

        val_rets = torch.stack(val_rets)
        self.board.add_scalars("val/returns", Stats(val_rets).asdict())

    def train_step(self):
        batch = []
        for _ in range(self.batch_size):
            step, done = next(self.env_iter)
            batch.append(step)
            if done or len(batch) >= self.batch_size:
                break

        batch = ListSeq.from_steps(batch)
        batch = TensorSeq.convert(batch).to(self.device)

        for _ in range(self.critic_steps):
            v, vt = self.critic(batch.obs), self.target_critic(batch.obs)
            gae_vt = []
            with torch.inference_mode():
                for step in reversed(range(len(v))):
                    if step == len(v) - 1:
                        cur_vt = vt[step]
                    else:
                        next_v = (1.0 - self.gae_lambda) * vt[
                            step + 1
                        ] + self.gae_lambda * gae_vt[-1]
                        cur_vt = batch.reward[step] + self.gamma * next_v

                    cont_f = 1.0 - batch.term[step].type_as(batch.obs)
                    cur_vt = cont_f * cur_vt

                    gae_vt.append(cur_vt)

            gae_vt.reverse()
            gae_vt = torch.stack(gae_vt)

            critic_loss = F.mse_loss(v, gae_vt)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.critic_polyak.step()

        logp = self.actor(batch.obs[:-1]).log_prob(batch.act)
        adv = (gae_vt - v)[:-1]
        vpg_loss = (logp * -adv.detach()).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        vpg_loss.backward()
        self.actor_opt.step()

        if self.should_log:
            self.board.add_scalar("train/critic_loss", critic_loss)
            self.board.add_scalar("train/vpg_loss", vpg_loss)
            self.board.add_scalars("train/logp", Stats(logp).asdict())
