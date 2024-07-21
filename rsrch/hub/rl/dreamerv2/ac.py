from contextlib import contextmanager
from functools import cached_property, partial, wraps
from threading import Lock
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
from .amp import autocast


@contextmanager
def no_grad(enabled=True):
    if enabled:
        with torch.no_grad():
            yield
    else:
        yield


class Actor(nn.Module):
    def __init__(self, wm: wm.WorldModel, cfg: config.ActorCritic):
        super().__init__()
        self.wm = wm
        self.cfg = cfg

        self.lock = Lock()
        self.actor = nn.Sequential(
            (fc := nets.FullyConnected(wm.state_size, None, **cfg.actor)),
            nets.PolicyLayer(fc.out_features, wm.act_enc),
        )

    def forward(self, state: rssm.State):
        return self.actor(state.to_tensor())

    def initial(self, obs: Tensor) -> rssm.State:
        obs = self.wm.obs_enc(obs)

        state = self.wm.rssm.initial(obs.device)
        state = state.expand(obs.shape[0], *state.shape)

        state = self.wm.rssm.obs_step(
            state,
            torch.zeros(
                (obs.shape[0], self.wm.rssm.act_size),
                device=obs.device,
            ),
            obs,
        )

        return state

    def step(self, state: rssm.State, act: Tensor, next_obs: Tensor):
        act = self.wm.act_enc(act)
        next_obs = self.wm.obs_enc(next_obs)
        next_s = self.wm.rssm.obs_step(state, act, next_obs)
        return next_s.type_as(state)


class Agent(gym.vector.Agent):
    def __init__(
        self,
        actor: Actor,
        cfg: config.Agent,
        mode: Literal["prefill", "train", "eval"],
    ):
        super().__init__()
        self.actor = actor
        self.cfg = cfg
        self.mode = mode

        self._act_space = self.actor.wm.act_space
        self._state = None

    def reset(self, idxes, obs, info):
        state = self.actor.initial(obs)
        if self._state is None:
            self._state = state.clone()
        else:
            self._state[idxes] = state.type_as(self._state)

    def policy(self, obs):
        if self.mode == "prefill":
            return self._act_space.sample([obs.shape[0]])
        elif self.mode in ("train", "eval"):
            act_dist: D.Distribution = self.actor(self._state.to_tensor())
            if self.mode == "train":
                enc_act = act_dist.sample()
                noise = self.cfg.expl_noise
            elif self.mode == "eval":
                enc_act = act_dist.mode
                noise = self.cfg.eval_noise
            act = self.actor.wm.act_dec(enc_act)
            act = self._apply_noise(act, noise)
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
        self._act = act

    def observe(self, idxes, next_obs, term, trunc, info):
        next_s = self.actor.step(
            self._state[idxes],
            self._act[idxes],
            next_obs,
        )
        self._state[idxes] = next_s.type_as(self._state)


class ActorCritic(nn.Module):
    def __init__(
        self,
        wm: wm.WorldModel,
        cfg: config.ActorCritic,
    ):
        super().__init__()
        self.wm = wm
        self.cfg = cfg
        self.fwd_lock = Lock()

        self.actor = nn.Sequential(
            (fc := nets.FullyConnected(wm.state_size, None, **cfg.actor)),
            nets.PolicyLayer(fc.out_features, wm.act_enc),
        )

        self.critic = nets.BoxDecoder(
            in_features=wm.state_size,
            space=spaces.torch.Box(shape=(), dtype=torch.float32),
            **cfg.critic,
        )

        if cfg.target_critic is not None:
            self.target_critic = nets.BoxDecoder(
                in_features=wm.state_size,
                space=spaces.torch.Box(shape=(), dtype=torch.float32),
                **cfg.critic,
            )
            polyak.sync(self.critic, self.target_critic)
            self.update_target = polyak.Polyak(
                source=self.critic,
                target=self.target_critic,
                **cfg.target_critic,
            )
        else:
            self.target_critic = self.critic

        self.rew_norm = nets.StreamNorm(**cfg.rew_norm)

        self.opt_iter = 0

        if cfg.actor_grad == "auto":
            discrete = isinstance(wm.act_enc.space, spaces.torch.Discrete)
            self.actor_grad = "reinforce" if discrete else "dynamics"
        else:
            self.actor_grad = cfg.actor_grad
        self.actor_grad_mix = self._make_sched(cfg.actor_grad_mix)

        self.actor_ent = self._make_sched(cfg.actor_ent)

    def save(self, state_only=False):
        ckpt = {"state": self.state_dict()}
        if not state_only:
            ckpt = {
                **ckpt,
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
            }
        return ckpt

    def load(self, ckpt):
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
            self.opt.load_state_dict(ckpt["opt"])
        self.load_state_dict(ckpt["state"])

    @property
    def device(self):
        return next(self.parameters()).device

    @cached_property
    def opt(self):
        cfg = {**self.cfg.opt}

        cls = config.get_class(torch.optim, cfg["$type"])
        del cfg["$type"]

        actor, critic = cfg["actor"], cfg["critic"]
        del cfg["actor"]
        del cfg["critic"]

        common = cfg

        return cls(
            [
                {"params": self.actor.parameters(), **actor},
                {"params": self.critic.parameters(), **critic},
            ],
            **common,
        )

    @cached_property
    def scaler(self) -> torch.cpu.amp.GradScaler:
        return getattr(torch, self.device.type).amp.GradScaler()

    def _make_sched(self, cfg: float | dict):
        if isinstance(cfg, float):
            return sched.Constant(cfg)
        else:
            cfg = {**cfg}
            cls = getattr(sched, cfg["$type"])
            del cfg["$type"]
            return cls(**cfg)

    def train_step(self, init: rssm.State, is_terminal: Tensor):
        mets = {}

        if self.update_target is not None:
            self.update_target.step()

        with self.fwd_lock:
            if not self.training:
                self.train()

            with autocast(self.device):
                states, acts = self._imagine(init, self.cfg.horizon)
                feats = states.to_tensor()

            # For reinforce, we detach `target` variable, so requiring gradients on
            # any computation leading up to it will cause autograd to leak memory
            with no_grad(enabled=(self.actor_grad == "reinforce")):
                with autocast(self.device):
                    rew_dist = over_seq(self.wm.reward_dec)(feats)
                    reward = self.rew_norm(rew_dist.mode)

                    term_dist = over_seq(self.wm.term_dec)(feats)
                    term = term_dist.mean
                    term[0] = is_terminal.float()
                    gamma = self.cfg.gamma * (1.0 - term)

                    vt = over_seq(self.target_critic)(feats).mode
                    target = self._lambda_return(
                        reward[:-1], vt[:-1], gamma[:-1], bootstrap=vt[-1]
                    )

            with torch.no_grad():
                with autocast(self.device):
                    weight = torch.cat([torch.ones_like(gamma[:1]), gamma[:-1]])
                    weight = weight.cumprod(0)

            with autocast(self.device):
                policies = self.actor(feats[:-2].detach())
                if self.actor_grad == "dynamics":
                    objective = target[1:]
                elif self.actor_grad == "reinforce":
                    baseline = over_seq(self.target_critic)(feats[:-2]).mode
                    adv = (target[1:] - baseline).detach()
                    objective = adv * policies.log_prob(acts[:-1].detach())
                elif self.actor_grad == "both":
                    baseline = over_seq(self.target_critic)(feats[:-2]).mode
                    adv = (target[1:] - baseline).detach()
                    objective = adv * policies.log_prob(acts[:-1].detach())
                    mix = self.actor_grad_mix(self.opt_iter)
                    mets["actor_grad_mix"] = mix
                    objective = mix * target[1:] + (1.0 - mix) * objective

                ent_scale = self.actor_ent(self.opt_iter)
                mets["policy_ent_scale"] = ent_scale
                policy_ent = policies.entropy()
                mets["policy_ent"] = policy_ent.mean()
                objective = objective + ent_scale * policy_ent
                actor_loss = -(weight[:-2] * objective).mean()
                mets["actor_loss"] = actor_loss

                value_dist = over_seq(self.critic)(feats[:-1].detach())
                mets["value"] = value_dist.mean.mean()
                critic_loss = (weight[:-1] * -value_dist.log_prob(target)).mean()
                mets["critic_loss"] = critic_loss

        loss = actor_loss + critic_loss

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.clip_grad is not None:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                [*self.actor.parameters(), *self.critic.parameters()],
                max_norm=self.cfg.clip_grad,
            )
        self.scaler.step(self.opt)
        self.scaler.update()

        self.opt_iter += 1

        return mets

    def _imagine(self, init: rssm.State, horizon: int):
        state = init
        states, acts = [state], []

        for _ in range(horizon):
            policy: D.Distribution = self.actor(state.to_tensor().detach())
            act: Tensor = policy.rsample()
            acts.append(act)

            state = self.wm.rssm.img_step(state, act)
            states.append(state)

        states: rssm.State = torch.stack(states)
        acts: Tensor = torch.stack(acts)

        return states, acts

    def _lambda_return(
        self,
        reward: Tensor,
        value: Tensor,
        gamma: Tensor,
        bootstrap: Tensor,
    ):
        next_values = torch.cat((value[1:], bootstrap[None]), 0)
        inputs = reward + gamma * next_values * (1.0 - self.cfg.gae_lambda)

        returns, cur = [], bootstrap
        for t in reversed(range(len(inputs))):
            cur = inputs[t] + gamma[t] * self.cfg.gae_lambda * cur
            returns.append(cur)

        returns.reverse()
        return torch.stack(returns)


class Agent(gym.vector.Agent):
    def __init__(
        self,
        env_f: env.Factory,
        ac: ActorCritic,
        cfg: config.Agent,
        mode: Literal["prefill", "train", "eval"],
    ):
        super().__init__()
        self.env_f = env_f
        self.ac = ac
        self.cfg = cfg
        self.mode = mode

        self._act_space = self.ac.wm.act_space
        self._num_envs, self._state = None, None

    @staticmethod
    def inference(_func):
        @wraps(_func)
        def wrapper(self: "Agent", *args, **kwargs):
            with self.ac.fwd_lock, self.ac.wm.fwd_lock:
                if self.ac.training:
                    self.ac.eval()

                with torch.inference_mode():
                    with autocast(self.ac.device):
                        return _func(self, *args, **kwargs)

        return wrapper

    @inference
    def reset(self, idxes, obs, info):
        obs = self.env_f.move_obs(obs)
        enc_obs: Tensor = self.ac.wm.obs_enc(obs)
        device = self.ac.device

        if self._state is None:
            assert idxes == np.arange(len(idxes))
            self._num_envs = len(idxes)
            self._state = self.ac.wm.rssm.initial(device)
            self._state = self._state.expand(self._num_envs, *self._state.shape)
            self._state = self._state.clone()
        else:
            self._state[idxes] = self.ac.wm.rssm.initial(device)

        enc_act = torch.zeros(
            (len(idxes), self.ac.wm.rssm.act_size),
            device=device,
        )

        value = self.ac.wm.rssm.obs_step(self._state[idxes], enc_act, enc_obs)
        self._state[idxes] = value.type_as(self._state)

    @inference
    def policy(self, obs):
        if self.mode == "prefill":
            return self.env_f.env_act_space.sample([self._num_envs])
        elif self.mode in ("train", "eval"):
            act_dist: D.Distribution = self.ac.actor(self._state.to_tensor())
            if self.mode == "train":
                enc_act = act_dist.sample()
                noise = self.cfg.expl_noise
            elif self.mode == "eval":
                enc_act = act_dist.mode
                noise = self.cfg.eval_noise
            act = self.ac.wm.act_dec(enc_act)
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

    @inference
    def step(self, act):
        act = self.env_f.move_act(act, to="net")
        enc_act = self.ac.wm.act_enc(act)
        self._enc_act = enc_act

    @inference
    def observe(self, idxes, next_obs, term, trunc, info):
        enc_act = self._enc_act[idxes]
        next_obs = self.env_f.move_obs(next_obs)
        enc_next_obs = self.ac.wm.obs_enc(next_obs)

        value = self.ac.wm.rssm.obs_step(self._state[idxes], enc_act, enc_next_obs)
        self._state[idxes] = value.type_as(self._state)
