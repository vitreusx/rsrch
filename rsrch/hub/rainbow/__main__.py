from __future__ import annotations

import argparse
import math
import multiprocessing
import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from numbers import Number

import numpy as np
import ray
import torch
import torch.nn.functional as F
from ray.util.queue import Queue as RayQueue
from torch import Tensor
from tqdm.auto import tqdm

import rsrch.distributions as D
from rsrch import nn
from rsrch.exp.board.null import NullBoard
from rsrch.exp.board.tb import TensorBoard
from rsrch.exp.dir import ExpDir
from rsrch.exp.prof.null import NullProfiler
from rsrch.exp.prof.torch import TorchProfiler
from rsrch.rl import gym
from rsrch.rl.data import interact
from rsrch.rl.data.seq import store
from rsrch.rl.data.seq.buffer import MultiStepBuffer
from rsrch.rl.data.seq.data import NumpySeqBatch, PaddedSeqBatch
from rsrch.rl.data.transforms import ToTensorSeq
from rsrch.rl.gym.agents import EpsAgent, FromTensor, RandomAgent, Wrap
from rsrch.rl.utils import polyak
from rsrch.types import rq_tree
from rsrch.types.tensorlike import Tensorlike
from rsrch.utils import data, sched
from rsrch.utils.cron import Every, Never


class ValueDist(D.Distribution, Tensorlike):
    def __init__(
        self,
        v_min: Tensor | Number,
        v_max: Tensor | Number,
        N: int,
        index_rv: D.Categorical,
    ):
        Tensorlike.__init__(self, index_rv.shape)
        self.event_shape = torch.Size([])
        self.v_min = v_min
        self.v_max = v_max
        self.N = N

        self.index_rv: D.Categorical
        self.register("index_rv", index_rv)

    @property
    def device(self):
        return self.index_rv._param.device

    @property
    def dtype(self):
        return self.index_rv._param.dtype

    @property
    def probs(self):
        return self.index_rv.probs

    @property
    def logits(self):
        return self.index_rv.logits

    @property
    def log_probs(self):
        return self.index_rv.log_probs

    @property
    def mode(self):
        idx = self.index_rv._param.argmax(-1)
        t = idx / (self.N - 1)
        return self.v_min * (1.0 - t) + self.v_max * t

    @property
    def supp(self):
        if isinstance(self.v_min, Tensor):
            t = torch.linspace(0.0, 1.0, self.N, device=self.device, dtype=self.dtype)
            return torch.outer(self.v_min, 1.0 - t) + torch.outer(self.v_max, t)
        else:
            return torch.linspace(
                self.v_min, self.v_max, self.N, device=self.device, dtype=self.dtype
            )

    @property
    def mean(self):
        probs = self.index_rv.probs
        return (probs * self.supp).sum(-1)

    def sample(self, sample_shape: torch.Size = torch.Size()):
        idx = self.index_rv.sample(sample_shape)
        t = idx / (self.N - 1)
        return self.v_min * (1.0 - t) + self.v_max * t

    def argmax(self, dim=None, keepdim=False):
        return self.mean.argmax(dim, keepdim)

    def gather(self, dim: int, index: Tensor):
        dim = range(len(self.shape))[dim]
        index = index[..., None]
        index = index.expand(*index.shape[:-1], self.N)
        if self.index_rv._probs is not None:
            probs = self.index_rv._probs.gather(dim, index)
            new_rv = D.Categorical(probs=probs)
        else:
            logits = self.index_rv._logits.gather(dim, index)
            new_rv = D.Categorical(logits=logits)
        return ValueDist(self.v_min, self.v_max, self.N, new_rv)

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        onehot_rv = D.OneHotCategoricalST(
            probs=self.index_rv._probs,
            logits=self.index_rv._logits,
        )
        onehot = onehot_rv.rsample(sample_shape)
        grid = torch.linspace(
            self.v_min,
            self.v_max,
            self.N,
            dtype=onehot.dtype,
            device=onehot.device,
        )
        return (onehot * grid).sum(-1)

    def entropy(self):
        return self.index_rv.entropy()

    def __add__(self, dv):
        return self._with_supp(self.v_min + dv, self.v_max + dv)

    def _with_supp(self, new_v_min, new_v_max):
        return ValueDist(new_v_min, new_v_max, self.N, self.index_rv)

    def __radd__(self, dv):
        return self._with_supp(dv + self.v_min, dv + self.v_max)

    def __sub__(self, dv):
        return self._with_supp(self.v_min - dv, self.v_max - dv)

    def __mul__(self, scale):
        return self._with_supp(self.v_min * scale, self.v_max * scale)

    def __rmul__(self, scale: float):
        return self._with_supp(scale * self.v_min, scale * self.v_max)

    def __truediv__(self, div: float):
        return self._with_supp(self.v_min / div, self.v_max / div)

    @staticmethod
    def proj_kl_div(p: ValueDist, q: ValueDist):
        supp_p = p.supp.broadcast_to(*p.shape, p.N)  # [*B, N_p]
        supp_q = q.supp.broadcast_to(*q.shape, q.N)  # [*B, N_q]
        supp_p = supp_p.clamp(q.v_min[..., None], q.v_max[..., None])
        dz_q = (q.v_max - q.v_min) / (q.N - 1)
        dz_q = dz_q[..., None, None]
        supp_p, supp_q = supp_p[..., None, :], supp_q[..., None]
        t = (1 - (supp_p - supp_q).abs() / dz_q).clamp(0, 1)  # [*B, N_q, N_p]
        proj_probs = (t * p.probs[..., None, :]).sum(-1)  # [*B, N_q]
        kl_div = D.kl_divergence(D.Categorical(probs=proj_probs), q.index_rv)
        return kl_div


@dataclass
class DistCfg:
    enabled: bool = False
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51


class NatureEncoder(nn.Sequential):
    def __init__(self, obs_shape: torch.Size):
        in_channels, height, width = obs_shape
        assert (height, width) == (84, 84)
        self.out_features = 3136

        super().__init__(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )


class ImpalaSmall(nn.Sequential):
    def __init__(self, obs_shape: torch.Size):
        super().__init__(
            nn.Conv2d(obs_shape[0], 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
        )
        self.out_features = 1152


class ImpalaResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.main(x)


class ImpalaBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
            ImpalaResidual(out_channels),
            ImpalaResidual(out_channels),
        )


class ImpalaLarge(nn.Sequential):
    def __init__(self, obs_shape: torch.Size, model_size=1):
        super().__init__(
            ImpalaBlock(obs_shape[0], 16 * model_size),
            ImpalaBlock(16 * model_size, 32 * model_size),
            ImpalaBlock(32 * model_size, 64 * model_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((8, 8)),
            nn.Flatten(),
        )
        self.out_features = 4096 * model_size


class QHead(nn.Module):
    def __init__(self, in_features: int, num_actions: int, dist: DistCfg, hidden=256):
        super().__init__()
        self._num_actions = num_actions
        self._dist = dist

        self.value_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        if self._dist.enabled:
            adv_out = num_actions * self._dist.num_atoms
        else:
            adv_out = num_actions

        self.adv_stream = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, adv_out),
        )

    def forward(self, feat: Tensor) -> Tensor | ValueDist:
        value_out, adv_out = self.value_stream(feat), self.adv_stream(feat)

        if self._dist.enabled:
            value_out = value_out.reshape(-1, self._num_actions, 1)
            adv_out = adv_out.reshape(-1, self._num_actions, self._dist.num_atoms)
            logits = value_out + adv_out - adv_out.mean(-2, keepdim=True)
            return ValueDist(
                v_min=self._dist.v_min,
                v_max=self._dist.v_max,
                N=self._dist.num_atoms,
                index_rv=D.Categorical(logits=logits),
            )
        else:
            return value_out + adv_out - adv_out.mean(-1, keepdim=True)


class NoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma0: float,
        bias=True,
        factorized=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._sigma0 = sigma0
        self._bias = bias
        self._factorized = factorized

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.noisy_weight = nn.Parameter(torch.empty_like(self.weight))
        self.register_buffer("weight_eps", torch.empty_like(self.weight))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.noisy_bias = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_eps", torch.empty_like(self.bias))

        self.reset_weights()
        self.reset_noise()

    def reset_weights(self):
        s = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -s, s)
        nn.init.constant_(self.weight_eps, self._sigma0 * s)
        if self._bias:
            nn.init.uniform_(self.bias, -s, s)
            nn.init.constant_(self.bias_eps, self._sigma0 * s)

    def reset_noise(self):
        device, dtype = self.weight.device, self.weight.dtype

        eps_in = torch.randn(self.in_features, device=device, dtype=dtype)
        sign_in = self.eps_in.sign()
        eps_in.abs_().sqrt_().mul_(sign_in)

        eps_out = torch.randn(self.out_features, device=device, dtype=dtype)
        sign_out = self.weight_eps_v.sign()
        eps_out.abs_().sqrt_().mul_(sign_out)

        self.weight_eps.copy_(eps_out.outer(eps_in))
        if self._bias:
            self.bias_eps.copy_(eps_out)

    def forward(self, x):
        w = self.weight + self.noisy_weight * self.weight_eps
        if self._bias:
            b = self.bias + self.noisy_bias * self.bias_eps
            return F.linear(x, w, b)
        else:
            return F.linear(x, w)


class QAgent(nn.Module):
    def __init__(self, enc, q):
        super().__init__()
        self.enc = enc
        self.q = q

    def observe(self, obs):
        self._cur_obs = self.enc(obs[None])[0]

    def policy(self):
        with torch.inference_mode():
            q_values = self.q(self._cur_obs[None])[0]

        if isinstance(q_values, Tensor):
            return q_values.argmax(-1)
        elif isinstance(q_values, ValueDist):
            return q_values.mode.argmax(-1)
        else:
            raise NotImplementedError(type(q_values))

    def step(self, act):
        pass


def map_module_(module: nn.Module, f, fqdn=None):
    mod = f(fqdn, module)
    for name, child in module.named_children():
        fqdn_ = name if fqdn is None else f"{fqdn}.{name}"
        mod.add_module(name, map_module_(child, f, fqdn_))
    return mod


def main():
    device = "cuda"
    mixed_prec = False
    gamma = 0.99
    dist = DistCfg()
    total_steps = int(10e6)
    batch_size = 256
    replay_ratio = 8
    env_steps_per_batch = batch_size // replay_ratio
    prioritized = True
    prio_omega, prio_beta0, prio_eps = 0.7, 0.4, 1e-8
    noisy = False
    noisy_sigma0 = 0.5
    apply_sn = True
    prefill_steps = int(25e3)

    def make_env():
        env = gym.wrappers.AtariPreprocessing(
            gym.make("ALE/Alien-v5", frameskip=4),
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
            noop_max=30,
        )
        env = gym.wrappers.FrameStack(env, 4)
        env = gym.wrappers.TransformReward(env, lambda r: min(max(r, -1.0), 1.0))
        env = gym.wrappers.TimeLimit(env, int(108e3))
        return env

    train_env = make_env()
    val_env = make_env()

    buffer = MultiStepBuffer(int(1e6), 3)
    ds = data.Indexed(buffer)

    if prioritized:
        sampler = data.WeightedInfiniteSampler(ds, buffer.capacity)
        max_prio = rq_tree.RangeQueryTree(buffer.capacity, max, -np.inf)
        min_prio = rq_tree.RangeQueryTree(buffer.capacity, min, np.inf)
        prio_beta = sched.Linear(prio_beta0, 1.0, total_steps)
    else:
        sampler = data.InfiniteSampler(ds, shuffle=True)

    def collate_fn(batch):
        idxes, batch = zip(*batch)
        idxes = np.asarray(idxes)
        batch = NumpySeqBatch.collate_fn(batch)
        batch = PaddedSeqBatch.from_numpy(batch)
        batch = batch.to(dtype=torch.float32)

        batch.obs = batch.obs / 255.0
        return idxes, batch

    loader = data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    batch_iter = iter(loader)

    env_steps = 0
    pbar = tqdm(total=total_steps)

    prefill = interact.steps_ex(
        env=train_env,
        agent=RandomAgent(train_env),
        max_steps=prefill_steps,
    )
    for step, done in prefill:
        idx = buffer.add(step, done)
        env_steps += 1
        if prioritized and idx is not None:
            sampler.update(idx, 1.0)
            max_prio[idx] = 1.0
            min_prio[idx] = 1.0
        pbar.update()

    assert isinstance(val_env.observation_space, gym.spaces.Box)
    assert isinstance(val_env.action_space, gym.spaces.Discrete)

    def apply_sn_(name, attr):
        if isinstance(attr, nn.Conv2d):
            attr = nn.utils.spectral_norm(attr)
        return attr

    obs_shape = torch.Size(train_env.observation_space.shape)
    enc = ImpalaLarge(obs_shape, model_size=2)
    # enc = ImpalaSmall(obs_shape)
    if apply_sn:
        map_module_(enc, apply_sn_)
    enc = enc.to(device)

    def linear_to_noisy(name, attr):
        if isinstance(attr, nn.Linear):
            return NoisyLinear(
                in_features=attr.in_features,
                out_features=attr.out_features,
                sigma0=noisy_sigma0,
                bias=attr.bias is not None,
            )
        return attr

    def make_q():
        num_actions = train_env.action_space.n
        q = QHead(enc.out_features, num_actions, dist)
        if noisy:
            map_module_(q, linear_to_noisy)
        return q

    q = make_q().to(device)
    target_q = make_q().to(device)
    polyak.copy(q, target_q)

    # q_opt = torch.optim.Adam(q.parameters(), lr=6.25e-5, eps=1.25e-4)
    opt_params = [*q.parameters(), *enc.parameters()]
    opt = torch.optim.Adam(opt_params, lr=3e-4, eps=5e-3 / batch_size)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_prec)
    q_polyak = polyak.Polyak(q, target_q, every=int(32e3))

    autocast = lambda: torch.autocast(
        device_type=device,
        dtype=torch.bfloat16,
        enabled=mixed_prec,
    )

    def make_agent(eps_fn):
        agent = EpsAgent(QAgent(enc, q), RandomAgent(train_env), eps_fn)
        agent = Wrap(agent, obs_fn=lambda obs: obs / 255.0)
        agent = FromTensor(agent, train_env, device)
        return agent

    if not noisy:
        eps_sched = sched.Linear(1.0, 0.01, total_steps / 10)
        train_agent = make_agent(lambda: eps_sched(env_steps))
    else:
        train_agent = make_agent(lambda: 0.0)
    val_agent = make_agent(lambda: 0.0)
    env_inter = interact.steps_ex(train_env, train_agent)

    exp_dir = ExpDir("runs/rainbow")

    profile = False
    if profile:
        prof = TorchProfiler(
            schedule=TorchProfiler.schedule(3, 2, 5, 4),
            on_trace=TorchProfiler.on_trace_fn(exp_dir),
            use_cuda=(device == "cuda"),
        )
    else:
        prof = NullProfiler()

    log = True
    if log:
        board = TensorBoard(root_dir=exp_dir / "board", step_fn=lambda: env_steps)
        should_log = Every(lambda: env_steps, int(256))
    else:
        board = NullBoard()
        should_log = Never()

    def val_epoch():
        val_rs = []
        for _ in range(16):
            val_ep = interact.one_episode(val_env, val_agent)
            val_r = sum(val_ep.reward)
            val_rs.append(val_r)
        board.add_scalar("val/returns", np.median(val_rs))

    val_epoch = Every(lambda: env_steps, int(25e3), val_epoch)

    def train_step():
        for _ in range(env_steps_per_batch):
            step, done = next(env_inter)
            nonlocal env_steps
            env_steps += 1
            pbar.update()
            q_polyak.step()
            idx = buffer.add(step, done)
            if prioritized and idx is not None:
                max_prio[idx] = max_prio.total
                sampler.update(idx, max_prio.total)

        idxes, batch = next(batch_iter)
        batch = batch.to(device)

        with autocast():
            with torch.no_grad():
                z = enc(batch.obs[-1])
                act = q(z).argmax(-1)
                target = target_q(z).gather(-1, act[..., None]).squeeze(-1)
                target = (1.0 - batch.term.to(dtype=target.dtype)) * target
                R = sum(batch.reward[i] * gamma**i for i in range(len(batch.reward)))
                target = R + gamma ** len(batch.reward) * target

        with autocast():
            pred = q(enc(batch.obs[0])).gather(-1, batch.act[0][..., None]).squeeze(-1)
            if isinstance(target, ValueDist):
                q_loss = ValueDist.proj_kl_div(pred, target)
            elif isinstance(target, Tensor):
                q_loss = F.mse_loss(pred, target)
            else:
                raise NotImplementedError(type(target))

            if prioritized:
                prio = q_loss.detach().cpu().numpy()  # sqrt?
                cur_prio_beta = prio_beta(env_steps)
                imp_w = (sampler.prio[idxes] / min_prio.total) ** (-cur_prio_beta)
                imp_w = torch.as_tensor(imp_w, dtype=q_loss.dtype, device=q_loss.device)
                q_loss = (imp_w * q_loss).mean()
            else:
                q_loss = q_loss.mean()

        opt.zero_grad(set_to_none=True)
        scaler.scale(q_loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(opt_params, 10.0)
        scaler.step(opt)
        scaler.update()

        if prioritized:
            prio = prio**prio_omega + prio_eps
            sampler.update(idxes, prio)
            max_prio[idxes] = prio
            min_prio[idxes] = prio

        if should_log:
            board.add_scalar("train/loss", q_loss.item())
            board.add_scalar("train/agent_eps", eps_sched(env_steps))

    while env_steps < total_steps:
        val_epoch()
        with prof.profile("train_step"):
            train_step()


if __name__ == "__main__":
    main()
