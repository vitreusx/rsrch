import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributions as D
from torch import Tensor, nn, optim

from rsrch.exp import tensorboard
from rsrch.exp.pbar import ProgressBar

from . import env


@dataclass
class Config:
    seed: int = 1
    device: str = "cuda"
    env_id: str = "Ant-v4"
    total_steps: int = int(1e6)
    lr: float = 3e-4
    num_envs: int = 1
    env_batch: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 64
    opt_iters: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


def layer_init(layer, std=nn.init.calculate_gain("relu"), bias=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = D.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.device)

    env_cfg = env.Config(type="gym", gym=env.gym.Config(env_id=cfg.env_id))
    env_f = env.make_factory(env_cfg, device)
    envs = env_f.vector_env(cfg.num_envs)

    steps_per_batch = cfg.env_batch // cfg.num_envs
    num_epochs = cfg.total_steps // cfg.env_batch

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    if cfg.anneal_lr:
        lr_sched = optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, num_epochs)

    # ALGO Logic: Storage setup
    batch_shape = (steps_per_batch, cfg.num_envs)
    obs = torch.zeros(batch_shape + env_f.obs_space.shape).to(device)
    actions = torch.zeros(batch_shape + env_f.act_space.shape).to(device)
    logprobs = torch.zeros(batch_shape).to(device)
    rewards = torch.zeros(batch_shape).to(device)
    dones = torch.zeros(batch_shape).to(device)
    values = torch.zeros(batch_shape).to(device)
    ep_rets = np.zeros([cfg.num_envs])

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = env_f.move_obs(next_obs)
    next_done = torch.zeros(cfg.num_envs).to(device)

    exp = tensorboard.Experiment(project="hydra")
    env_step = 0
    exp.register_step("env_step", lambda: env_step, default=True)
    pbar = ProgressBar(desc="Hydra", total=cfg.total_steps)

    for _ in range(num_epochs):
        for step in range(steps_per_batch):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            action = env_f.move_act(action, to="env")
            action = action.clip(env_f._np_act_space.low, env_f._np_act_space.high)
            next_obs, reward, terminations, truncations, infos = envs.step(action)

            next_done = np.logical_or(terminations, truncations)
            ep_rets += reward
            for env_idx in range(cfg.num_envs):
                if next_done[env_idx]:
                    exp.add_scalar("train/ep_ret", ep_rets[env_idx])
                    ep_rets[env_idx] = 0
                env_step += 1
                pbar.update()

            next_done = torch.tensor(next_done).float().to(device)
            rew = torch.tensor(reward).to(device)
            rewards[step] = rew
            next_obs = env_f.move_obs(next_obs)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(steps_per_batch)):
                if t == steps_per_batch - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + env_f.obs_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env_f.act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.env_batch)
        clipfracs = []
        for _ in range(cfg.opt_iters):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.env_batch, cfg.batch_size):
                end = start + cfg.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        exp.add_scalar("train/value_loss", v_loss.item())
        exp.add_scalar("train/policy_loss", pg_loss.item())
        exp.add_scalar("train/ent", entropy_loss.item())
        exp.add_scalar("train/clip_frac", np.mean(clipfracs))

        lr_sched.step()
