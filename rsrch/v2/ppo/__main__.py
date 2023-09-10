import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn, Tensor
import rsrch.distributions as D
from rsrch.exp.board.wandb import Wandb
from rsrch.exp.dir import ExpDir
from rsrch.exp.pbar import ProgressBar
from rsrch.exp.vcs import WandbVCS
from rsrch.rl import gym
import rsrch.nn.dist_head as dh
from rsrch.rl.data.buffer import OnlineBuffer
from rsrch.rl.data.data import StepBatch
from rsrch.rl.data import rollout
from rsrch.rl.utils.decorr import decorrelate
from rsrch.rl.utils.make_env import EnvFactory
from rsrch.utils import config, cron, stats
from . import config
from .config import Config


T = gym.spaces.transforms


class Encoder(nn.Sequential):
    def __init__(self, env_spec: gym.EnvSpec):
        if isinstance(env_spec.observation_space, gym.spaces.TensorImage):
            num_channels = env_spec.observation_space.shape[0]
            super().__init__(
                nn.Conv2d(num_channels, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
            )
            self.enc_dim = 512

        elif isinstance(env_spec.observation_space, gym.spaces.TensorBox):
            obs_dim = int(np.prod(env_spec.observation_space.shape))
            super().__init__(
                nn.Flatten(),
                nn.Linear(obs_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )
            self.enc_dim = 64

        else:
            raise ValueError(type(env_spec.observation_space))


class CriticHead(nn.Sequential):
    def __init__(self, enc_dim: int):
        super().__init__(
            nn.Linear(enc_dim, 1),
            nn.Flatten(0),
        )


class ActorHead(nn.Module):
    def __init__(self, env_spec: gym.EnvSpec, enc_dim: int):
        super().__init__()

        if isinstance(env_spec.action_space, gym.spaces.TensorDiscrete):
            num_actions = int(env_spec.action_space.n)
            self.net = dh.Categorical(enc_dim, num_actions)
        elif isinstance(env_spec.action_space, gym.spaces.TensorBox):
            self.net = dh.Normal(enc_dim, env_spec.action_space.shape)
        else:
            raise ValueError(type(env_spec.action_space))

    def forward(self, z):
        return self.net(z)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)


class ActorCritic(nn.Module):
    def __init__(self, env_spec: gym.EnvSpec, share_enc=False, custom_init=False):
        super().__init__()
        self._share_enc = share_enc

        if self._share_enc:
            self.enc = Encoder(env_spec)
            self.actor_head = ActorHead(env_spec, self.enc.enc_dim)
            self.critic_head = CriticHead(self.enc.enc_dim)
        else:
            actor_enc = Encoder(env_spec)
            actor_head = ActorHead(env_spec, actor_enc.enc_dim)
            self.actor = nn.Sequential(actor_enc, actor_head)
            critic_enc = Encoder(env_spec)
            critic_head = CriticHead(critic_enc.enc_dim)
            self.critic = nn.Sequential(critic_enc, critic_head)

        if custom_init:
            self._custom_init()

    def _custom_init(self):
        if self._share_enc:
            self.enc.apply(layer_init)
            self.actor_head.apply(lambda x: layer_init(x, std=1e-2))
            self.critic_head.apply(lambda x: layer_init(x, std=1.0))
        else:
            self.actor[0].apply(layer_init)
            self.actor[1].apply(lambda x: layer_init(x, std=1e-2))
            self.critic[0].apply(layer_init)
            self.critic[1].apply(lambda x: layer_init(x, std=1.0))

    def forward(self, state: Tensor, values=True):
        if self._share_enc:
            z = self.enc(state)
            policy = self.actor_head(z)
            value = self.critic_head(z) if values else None
        else:
            policy = self.actor(state)
            value = self.critic(state) if values else None
        return policy, value


def gae_adv_est(r, v, gamma, gae_lambda):
    delta = (r + gamma * v[1:]) - v[:-1]
    adv = [delta[-1]]
    for t in reversed(range(1, len(r))):
        adv.append(delta[t - 1] + gamma * gae_lambda * adv[-1])
    adv.reverse()
    adv = torch.stack(adv)
    ret = v[:-1] + adv
    return adv, ret


class EnvFactory:
    def __init__(self, cfg: config.Env, record_stats=True):
        self.cfg = cfg
        self.record_stats = record_stats

    def _base_env(self):
        if self.cfg.type == "atari":
            atari = self.cfg.atari
            env = gym.make(self.cfg.name, frameskip=1)

            if self.record_stats:
                env = gym.wrappers.RecordEpisodeStatistics(env)

            env = gym.wrappers.AtariPreprocessing(
                env=env,
                frame_skip=atari.frame_skip,
                screen_size=atari.screen_size,
                terminal_on_life_loss=atari.term_on_life_loss,
                grayscale_obs=atari.grayscale,
                grayscale_newaxis=(atari.frame_stack is None),
                scale_obs=False,
                noop_max=atari.noop_max,
            )

            if atari.fire_reset:
                if "FIRE" in env.unwrapped.get_action_meanings():
                    env = gym.wrappers.FireResetEnv(env)

            channels_last = True
            if atari.frame_stack is not None and atari.frame_stack > 1:
                env = gym.wrappers.FrameStack(env, atari.frame_stack)
                env = gym.wrappers.Apply(env, np.array)
                channels_last = False

            env = gym.wrappers.Apply(
                env,
                T.BoxAsImage(
                    env.observation_space,
                    channels_last=channels_last,
                ),
            )

        else:
            env = gym.make(self.cfg.name)

        if self.cfg.time_limit is not None:
            env = gym.wrappers.TimeLimit(env, self.cfg.time_limit)

        return env

    def val_env(self):
        env = self._base_env()
        env = gym.wrappers.ToTensor(env)
        return env

    def train_env(self):
        env = self._base_env()

        if self.cfg.type == "atari":
            atari = self.cfg.atari
            if atari.episodic_life:
                env = gym.wrappers.EpisodicLifeEnv(env)

        if self.cfg.reward in ("keep", None):
            rew_f = lambda r: r
        elif self.cfg.reward == "sign":
            env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
            env.reward_range = (-1, 1)
        elif isinstance(self.cfg.reward, tuple):
            r_min, r_max = self.cfg.reward
            rew_f = lambda r: np.clip(r, r_min, r_max)
            env = gym.wrappers.TransformReward(env, rew_f)
            env.reward_range = (r_min, r_max)

        env = gym.wrappers.ToTensor(env)
        return env


def main():
    cfg = config.from_args(
        cls=Config,
        defaults=Path(__file__).parent / "config.yml",
        presets=Path(__file__).parent / "presets.yml",
    )

    device = torch.device(cfg.device)

    env_f = EnvFactory(cfg.env, record_stats=True)
    make_val_env = env_f.val_env
    make_train_env = env_f.train_env

    def make_venv(env_fns):
        if len(env_fns) == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv2(env_fns, cfg.env_workers)

    val_env = make_venv([make_val_env] * cfg.val_episodes)
    train_env_fns = [make_train_env] * cfg.train_envs
    if cfg.decorrelate:
        train_env_fns, states = decorrelate(train_env_fns)
        train_env = make_venv(train_env_fns)
        states = gym.vector.utils.merge_vec_infos(states)
        init = states["obs"], states["info"]
    else:
        train_env = make_venv(train_env_fns)
        init = None

    class ACAgent(gym.Agent):
        def __init__(self, ac: ActorCritic, env_spec: gym.EnvSpec):
            self.ac = ac
            self._model_dev = next(ac.parameters()).device
            self._env_dev = env_spec.observation_space.device

        @torch.inference_mode()
        def policy(self, obs: tuple[Tensor]):
            obs = torch.stack(obs)
            act_rv, _ = self.ac(obs.to(self._model_dev), values=False)
            return tuple(act_rv.sample().to(self._env_dev))

    env_spec = gym.EnvSpec(make_train_env())

    actor_critic = ActorCritic(
        env_spec, share_enc=cfg.share_encoder, custom_init=cfg.custom_init
    ).to(device)
    opt = torch.optim.Adam(actor_critic.parameters(), lr=cfg.lr, eps=cfg.opt_eps)

    train_agent = ACAgent(actor_critic, env_spec)
    val_agent = ACAgent(actor_critic, env_spec)

    ep_ids = [None for _ in range(train_env.num_envs)]
    buf = OnlineBuffer()
    env_iter = iter(rollout.steps(train_env, train_agent, init=init))

    env_step = 0
    exp_dir = ExpDir("runs/ppo")
    board = Wandb(project="ppo", step_fn=lambda: env_step)
    pbar = ProgressBar(total=cfg.total_steps)
    vcs = WandbVCS()
    vcs.save()

    should_log = cron.Every(lambda: env_step, cfg.log_every)
    should_val = cron.Every(lambda: env_step, cfg.val_every)

    train_ep_ret = stats.RunningAvg(last_k=8)

    while env_step < cfg.total_steps:
        if should_val:
            val_returns = []
            val_iter = rollout.episodes(
                val_env, val_agent, max_episodes=cfg.val_episodes
            )
            for _, ep in val_iter:
                val_returns.append(sum(ep.reward))
            board.add_scalar("val/returns", np.mean(val_returns))

        buf.reset()
        for _ in range(cfg.steps_per_epoch * cfg.train_envs):
            env_idx, step = next(env_iter)
            ep_ids[env_idx] = buf.push(ep_ids[env_idx], step)
            if "episode" in step.info:
                board.add_scalar("train/ep_ret", step.info["episode"]["r"])
            env_step += 1
            pbar.update()

        train_eps = [*buf.values()]
        obs, act, logp, adv, ret, value = [], [], [], [], [], []

        with torch.no_grad():
            for ep in train_eps:
                ep_obs = torch.stack(ep.obs).to(device)
                obs.append(ep_obs[:-1])
                ep_policy, ep_value = actor_critic(ep_obs)
                if ep.term:
                    ep_value[-1] = 0.0
                value.append(ep_value[:-1])
                ep_reward = torch.tensor(ep.reward).type_as(ep_value)
                ep_adv, ep_ret = gae_adv_est(
                    ep_reward, ep_value, cfg.gamma, cfg.gae_lambda
                )
                adv.append(ep_adv)
                ret.append(ep_ret)
                ep_act = torch.stack(ep.act).to(device)
                act.append(ep_act)
                ep_logp = ep_policy[:-1].log_prob(ep_act)
                logp.append(ep_logp)

        cat_ = [torch.cat(x) for x in (obs, act, logp, adv, ret, value)]
        obs, act, logp, adv, ret, value = cat_

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(len(act))
            for idxes in perm.split(cfg.update_batch):
                new_policy, new_value = actor_critic(obs[idxes])
                new_logp = new_policy.log_prob(act[idxes])
                log_ratio = new_logp - logp[idxes]
                ratio = log_ratio.exp()

                adv_ = adv[idxes]
                if cfg.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / (adv_.std() + 1e-8)

                t1 = -adv_ * ratio
                t2 = -adv_ * ratio.clamp(1 - cfg.clip_coeff, 1 + cfg.clip_coeff)
                policy_loss = torch.max(t1, t2).mean()

                if cfg.clip_vloss:
                    clipped_v = value[idxes] + (new_value - value[idxes]).clamp(
                        -cfg.clip_coeff, cfg.clip_coeff
                    )
                    v_loss1 = (new_value - ret[idxes]).square()
                    v_loss2 = (clipped_v - ret[idxes]).square()
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    v_loss = 0.5 * (new_value - ret[idxes]).square().mean()

                ent_loss = -new_policy.entropy().mean()

                loss = policy_loss + cfg.ent_coeff * ent_loss + cfg.vf_coeff * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.clip_grad is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        actor_critic.parameters(), cfg.clip_grad
                    )
                opt.step()

        if should_log:
            board.add_scalar("train/loss", loss)
            board.add_scalar("train/policy_loss", policy_loss)
            board.add_scalar("train/v_loss", v_loss)
            board.add_scalar("train/ent_loss", ent_loss)
            board.add_scalar("train/mean_v", value.mean())


if __name__ == "__main__":
    main()
