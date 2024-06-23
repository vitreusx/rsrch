from contextlib import contextmanager
from functools import cached_property, partial
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn.utils import over_seq, safe_mode
from rsrch.rl import gym
from rsrch.rl.utils import polyak
from rsrch.utils import sched

from . import config, env, nets, rssm, wm


class ActorCritic(nn.Module):
    def __init__(
        self,
        wm: wm.WorldModel,
        cfg: config.ActorCritic,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.wm = wm
        self.cfg = cfg

        fc = nets.FC(wm.state_size, None, **cfg.actor)
        self.actor = nn.Sequential(
            fc,
            nets.ActorHead(fc.out_features, wm.act_enc),
        )

        self.critic = nn.Sequential(
            nets.FC(wm.state_size, 1, **cfg.critic),
            nn.Flatten(0),
        )

        if cfg.target_critic is not None:
            self.target_critic = nn.Sequential(
                nets.FC(wm.state_size, 1, **cfg.critic),
                nn.Flatten(0),
            )
            polyak.sync(self.critic, self.target_critic)
            self.update_target = polyak.Polyak(
                self.critic, self.target_critic, **cfg.target_critic
            )
        else:
            self.target_critic = self.critic
            self.update_target = None

        self.actor_opt = self._make_optim(cfg.actor_opt)(self.actor.parameters())
        self.critic_opt = self._make_optim(cfg.critic_opt)(self.critic.parameters())

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)

        self.opt_iter = 0

        if cfg.actor_grad == "auto":
            discrete = isinstance(wm.act_enc.space, spaces.torch.Discrete)
            self.actor_grad = "reinforce" if discrete else "dynamics"
        else:
            self.actor_grad = cfg.actor_grad
        self.actor_grad_mix = self._make_sched(cfg.actor_grad_mix)

        self.actor_ent = self._make_sched(cfg.actor_ent)

    def _make_optim(self, cfg: dict):
        cfg = {**cfg}
        cls = getattr(torch.optim, cfg["$type"])
        del cfg["$type"]
        return partial(cls, **cfg)

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, float):
            return sched.Constant(cfg)
        else:
            cfg = {**cfg}
            cls = getattr(sched, cfg["$type"])
            del cfg["$type"]
            return cls(**cfg)

    @cached_property
    def actor_scaler(self):
        return getattr(torch, self.device.type).amp.GradScaler()

    @cached_property
    def critic_scaler(self):
        return getattr(torch, self.device.type).amp.GradScaler()

    @cached_property
    def device(self):
        return next(self.parameters()).device

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def opt_step(self, init: rssm.State, is_terminal: Tensor):
        mets = {}

        with self.autocast():
            states, acts, policies = self._imagine(init, self.cfg.horizon)
            state_t = states.to_tensor()

            rew_dist = over_seq(self.wm.reward_dec)(state_t)
            reward = self.rew_norm(rew_dist.mode)

            term_dist = over_seq(self.wm.term_dec)(state_t)
            term = term_dist.mean
            term[0] = is_terminal.float()
            gamma = self.cfg.gamma * (1.0 - term)

            weight = torch.cat([torch.ones_like(gamma[:1]), gamma[:-1]]).cumprod(0)

        with torch.no_grad():
            with self.autocast():
                vt = over_seq(self.target_critic)(state_t)

        with self.autocast():
            target = self._gae_returns(reward[:-1], vt, gamma[:-1])

            policies = policies[:-1]
            if self.actor_grad == "dynamics":
                objective = target[1:]
            elif self.actor_grad == "reinforce":
                baseline = vt[:-2]
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(acts[:-1].detach())
            elif self.actor_grad == "both":
                baseline = vt[:-2]
                adv = (target[1:] - baseline).detach()
                objective = adv * policies.log_prob(acts[:-1].detach())
                mix = self.actor_grad_mix(self.opt_iter)
                mets["actor_grad_mix"] = mix
                objective = mix * target[1:] + (1.0 - mix) * objective

            ent_scale = self.actor_ent(self.opt_iter)
            policy_ent = policies.entropy()
            objective = objective + ent_scale * policy_ent
            actor_loss = -(weight[:-2].detach() * objective).mean()

            mets["policy_ent"] = policy_ent.mean()
            mets["policy_ent_scale"] = ent_scale
            mets["actor_loss"] = actor_loss

        opt, scaler = self.actor_opt, self.actor_scaler
        opt.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward(retain_graph=True)
        if self.cfg.clip_grad is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self.cfg.clip_grad,
            )
        scaler.step(opt)
        scaler.update()

        with self.autocast():
            state_ = state_t.detach().clone()
            value = over_seq(self.critic)(state_[:-1])
            mets["value"] = value.mean()
            target_ = target.detach().clone()
            weight_ = weight.detach().clone()
            error = (value - target_).square()
            critic_loss = -(weight_[:-1] * error).mean()
            mets["critic_loss"] = critic_loss

        opt, scaler = self.critic_opt, self.critic_scaler
        opt.zero_grad(set_to_none=True)
        scaler.scale(critic_loss).backward()
        if self.cfg.clip_grad is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.cfg.clip_grad,
            )
        scaler.step(opt)
        scaler.update()

        self.opt_iter += 1
        if self.update_target is not None:
            self.update_target.step()

        return mets

    def _imagine(self, init: rssm.State, horizon: int):
        state = init
        states, acts, policies = [state], [], []

        for _ in range(horizon):
            policy: D.Distribution = self.actor(state.to_tensor().detach())
            policies.append(policy)

            act: Tensor = policy.rsample()
            acts.append(act)

            state_dist: rssm.StateDist = self.wm.rssm.img_step(state, act)
            state = state_dist.rsample()
            states.append(state)

        states: rssm.State = torch.stack(states)
        acts: Tensor = torch.stack(acts)
        policies: D.Distribution = torch.stack(policies)

        return states, acts, policies

    def _gae_returns(self, rewards: Tensor, values: Tensor, gammas: Tensor):
        inputs = rewards + gammas * values[1:] * (1.0 - self.cfg.gae_lambda)
        returns, cur = [], values[-1]
        for t in reversed(range(len(inputs))):
            cur = inputs[t] + gammas[t] * self.cfg.gae_lambda * cur
            returns.append(cur)
        returns.reverse()
        return torch.stack(returns)


class Agent(gym.vector.Agent, nn.Module):
    def __init__(
        self,
        env_f: env.Factory,
        ac: ActorCritic,
        cfg: config.Agent,
        mode: Literal["prefill", "train", "eval"],
    ):
        nn.Module.__init__(self)
        gym.vector.Agent.__init__(self)
        self.env_f = env_f
        self.ac = ac
        self.cfg = cfg
        self.mode = mode

        self._wm = ac.wm
        self._act_space = self._wm.act_space
        self._num_envs, self._state = None, None

    @contextmanager
    def autocast(self):
        with torch.autocast(device_type=self.ac.device.type, dtype=self.ac.dtype):
            with safe_mode(self):
                yield

    def reset(self, idxes, obs, info):
        with self.autocast():
            obs = self.env_f.move_obs(obs)
            enc_obs: Tensor = self._wm.obs_enc(obs)
            device, dtype = self.ac.device, self.ac.dtype

            if self._state is None:
                assert idxes == np.arange(len(idxes))
                self._num_envs = len(idxes)
                self._state = self._wm.rssm.initial(device, dtype)
                self._state = self._state.expand(self._num_envs, *self._state.shape)
                self._state = self._state.clone()
            else:
                self._state[idxes] = self._wm.rssm.initial(device, dtype)

            enc_act = torch.zeros(
                (len(idxes), self._wm.rssm.act_size),
                dtype=dtype,
                device=device,
            )
            is_first = torch.ones(
                [len(idxes)],
                dtype=torch.bool,
                device=device,
            )

            prior, post = self._wm.rssm.obs_step(
                self._state[idxes], enc_act, enc_obs, is_first
            )
            self._state[idxes] = post.sample().to(dtype)

    def policy(self, obs):
        if self.mode == "prefill":
            return self.env_f.env_act_space.sample([self._num_envs])
        elif self.mode in ("train", "eval"):
            with self.autocast():
                act_dist: D.Distribution = self.ac.actor(self._state.to_tensor())
                if self.mode == "train":
                    enc_act = act_dist.sample()
                    noise = self.cfg.expl_noise
                elif self.mode == "eval":
                    enc_act = act_dist.mode
                    noise = self.cfg.eval_noise
                act = self._wm.act_dec(enc_act)
                act = self._apply_noise(act, noise)
                act = self.env_f.move_act(act, to="env")
                return act

    def _apply_noise(self, act: Tensor, noise: float):
        if noise > 0.0:
            n = len(act)
            if isinstance(self._act_space, spaces.torch.Discrete):
                rand_act = self._act_space.sample((n,)).type_as(act)
                use_rand = (torch.rand(n) < noise).to(act.device)
                act = torch.where(use_rand, rand_act, act)
            elif isinstance(self._act_space, spaces.torch.Box):
                eps = torch.randn(self._act_space.shape).type_as(act)
                low = self._act_space.low.type_as(act)
                high = self._act_space.high.type_as(act)
                act = (act + noise * eps).clamp(low, high)
        return act

    def step(self, act):
        with self.autocast():
            act = self.env_f.move_act(act, to="net")
            enc_act = self._wm.act_enc(act)
            self._enc_act = enc_act

    def observe(self, idxes, next_obs, term, trunc, info):
        with self.autocast():
            enc_act = self._enc_act[idxes]
            next_obs = self.env_f.move_obs(next_obs)
            next_obs = self._wm.obs_enc(next_obs)

            prior, post = self._wm.rssm.obs_step(self._state[idxes], enc_act, next_obs)
            self._state[idxes] = post.sample().to(self.ac.dtype)
