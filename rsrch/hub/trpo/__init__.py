from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from itertools import count
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np
import scipy.signal
import torch
import torch.distributions as D
import torch.nn.functional as nn_F
from scipy.sparse.linalg import LinearOperator, cg
from torch import Tensor, autograd, nn
from tqdm.auto import tqdm

from rsrch.rl import agents, gym, wrappers
from rsrch.rl.data import EpisodeRollout, StepBatch, StepRollout
from rsrch.rl.data import transforms as T
from rsrch.rl.data.step import TensorStep
from rsrch.rl.spec import EnvSpec
from rsrch.utils import data
from rsrch.utils.board import Board
from rsrch.utils.copy import clone_module
from rsrch.utils.eval_ctx import eval_ctx
from rsrch.utils.exp_dir import ExpDir
from rsrch.vpg import rev_cumsum


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


class LinearOp:
    class ToNumpy(LinearOperator):
        def __init__(self, A: LinearOp):
            super().__init__(np.float32, tuple(A.shape))
            self.A = A

        def _matvec(self, x: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(x).to(self.A.device)
            return (self.A @ x).cpu().numpy()

    shape: torch.Size
    device: torch.device

    def __matmul__(self, x: Tensor) -> Tensor:
        ...

    def numpy(self):
        return self.ToNumpy(self)


def solve_cg(A: LinearOp, b: Tensor, niters: int, x: Optional[Tensor] = None) -> Tensor:
    """Solve Ax = b using conjugate gradients."""

    x, _ = cg(A.numpy(), b.cpu().numpy(), maxiter=niters)
    x = torch.from_numpy(x).type_as(b)

    # if x is None:
    #     n, _ = A.shape
    #     x = torch.zeros(n, dtype=b.dtype, device=b.device)

    # r_k = b - A @ x
    # rk_dot = torch.dot(r_k, r_k)

    # p_k = r_k.clone()
    # for k in range(niters):
    #     Ap_k = A @ p_k
    #     alpha_k = rk_dot / (torch.dot(p_k, Ap_k) + 1e-8)
    #     x += alpha_k * p_k
    #     r_k -= alpha_k * Ap_k
    #     prev_rk, rk_dot = rk_dot, torch.dot(r_k, r_k)
    #     p_k = r_k + (rk_dot / prev_rk) * p_k

    return x


class Tensors:
    """A helper for accessing and mutating lists of tensors."""

    def __init__(self, tensors: Sequence[Tensor]):
        self.tensors = [*tensors]
        self._shapes = [x.shape for x in self.tensors]
        self._volumes = [int(np.prod(x.shape)) for x in self.tensors]
        self.total = sum(self._volumes)

    def __iter__(self):
        return iter(self.tensors)

    @property
    def flat(self):
        """Convert tensors to a 1d tensor."""
        return torch.cat([x.flatten() for x in self.tensors])

    @flat.setter
    def flat(self, theta: Tensor):
        """Set the tensors from a 1d tensor."""
        tensors = theta.split(self._volumes)
        tensors = [x.reshape(shape) for x, shape in zip(tensors, self._shapes)]
        for source, target in zip(tensors, self.tensors):
            if isinstance(target, torch.nn.Parameter):
                target = target.data
            target.copy_(source)

    def zero_grad(self):
        for x in self.tensors:
            x.grad = None

    @property
    def grads(self):
        """Get network gradients as a 1d tensor."""
        return Tensors([x.grad for x in self.tensors])


class MeanKL:
    """A helper class for computing E_s KL(p(-|s) || q(-|s)), as well as the Hessian-vector product w.r.t. p's parameters."""

    class Hess(LinearOp):
        """Hessian of the mean KL divergence, or rather it's LinearOp form.\n
        NOTE: if p or q change, the hessian must be recreated."""

        def __init__(self, mean_kl: MeanKL):
            self.mean_kl = mean_kl
            self.p_params = Tensors(self.mean_kl.p.parameters())
            mean_kl_val = self.mean_kl()
            jac = autograd.grad(mean_kl_val, self.p_params, create_graph=True)
            self.g = Tensors(jac).flat

            num_params = self.p_params.total
            self.shape = torch.Size([num_params, num_params])
            self.device = self.g.device

        def __matmul__(self, x: Tensor) -> Tensor:
            hess = autograd.grad(torch.dot(self.g, x), self.p_params, retain_graph=True)
            return Tensors(hess).flat

    def __init__(self, p: Policy, q: Policy, obs: Tensor, batch_size: int):
        self.p, self.q = p, q
        self.obs = obs
        self.batch_size = batch_size

    def __call__(self):
        kl_values = []
        for off in range(0, len(self.obs), self.batch_size):
            batch_obs = self.obs[off : off + self.batch_size]
            batch_kl = D.kl_divergence(self.p(batch_obs), self.q(batch_obs))
            kl_values.append(batch_kl)
        return torch.cat(kl_values).mean()

    @property
    def hess(self):
        return MeanKL.Hess(self)


class TRPOTrainer:
    def __init__(self):
        self.env_steps_per_update = int(2**10)
        self.batch_size = 128
        assert self.env_steps_per_update % self.batch_size == 0
        self.device = torch.device("cuda")
        self.val_every_epoch = 64
        self.val_episodes = 16
        self.gamma = 0.99
        self.td_lambda = 0.97
        self.cg_iters = 40
        self.delta = 0.01
        self.ls_iters = 10
        self.ls_step = 0.8
        self.value_optim_iters = 40

    def train(self, trpo: Agent, trpo_data: Data):
        self.trpo, self.trpo_data = trpo, trpo_data

        self.init_envs()
        self.init_data()
        self.init_model()
        self.init_extras()
        self.loop()

    def init_envs(self):
        self.val_env = self.trpo_data.val_env(self.device)
        self.train_env = self.trpo_data.train_env(self.device)

    def init_data(self):
        self.ep_iter = None

    def init_model(self):
        self.trpo = self.trpo.to(self.device)
        self.pi = self.trpo.pi
        self.pi_params = Tensors(self.pi.parameters())
        self.aux_pi: Policy = self.pi.clone()
        self.aux_pi_params = Tensors(self.aux_pi.parameters())
        self.v_optim = torch.optim.Adam(self.trpo.V.parameters(), lr=1e-3)

    def init_extras(self):
        self.exp_dir = ExpDir()
        self.board = Board(root_dir=self.exp_dir / "board")
        self.pbar = tqdm(desc="TRPO")

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

            val_ep_rollout = EpisodeRollout(cur_env, self.trpo, num_episodes=1)
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
                self._steps = StepRollout(self.train_env, self.trpo, num_episodes=1)
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
                with eval_ctx(self.trpo.V):
                    ep_final_v = self.trpo.V(step.next_obs.unsqueeze(0)).item()
            final_v.append(ep_final_v)

            # Reset the episode iterator.
            self.ep_iter = None

        obs, act = torch.stack(obs), torch.stack(act)

        # Compute value estimates and log-probabilities of selected actions
        def compute_v():
            v = []
            for idx in range(0, self.env_steps_per_update, self.batch_size):
                batch_obs = obs[idx : idx + self.batch_size]
                batch_v = self.trpo.V(batch_obs)
                v.append(batch_v)
            return torch.cat(v)

        with torch.no_grad():
            v = compute_v()

        logp = []
        for idx in range(0, self.env_steps_per_update, self.batch_size):
            batch_obs = obs[idx : idx + self.batch_size]
            batch_act = act[idx : idx + self.batch_size]
            batch_logp = self.pi(batch_obs).log_prob(batch_act)
            logp.append(batch_logp)
        logp = torch.cat(logp)

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

        # 1. Obtain VPG gradient
        vpg_pi_loss = -(adv * logp).mean()
        grads = autograd.grad(-vpg_pi_loss, self.pi_params)
        g = Tensors(grads).flat

        # 2. Compute x = H^{-1}g
        self.aux_pi.load_state_dict(self.pi.state_dict())
        mean_kl = MeanKL(self.aux_pi, self.pi, obs, self.batch_size)
        H = mean_kl.hess
        x = solve_cg(H, g, self.cg_iters)

        # 4. Perform backtracking line search
        s = torch.sqrt(2.0 * self.delta / (torch.dot(x, H @ x) + 1e-8)) * x
        old_theta, old_pi_score = self.pi_params.flat, adv.mean()

        def cond(alpha: float) -> bool:
            new_theta = old_theta + alpha * s
            self.aux_pi_params.flat = new_theta

            mean_kl_val = mean_kl()
            if mean_kl_val >= self.delta:
                return False

            ratios = []
            for off in range(0, len(obs), self.batch_size):
                idxes = slice(off, off + self.batch_size)
                old_logp, batch_obs, batch_act = logp[idxes], obs[idxes], act[idxes]
                new_logp = self.aux_pi(batch_obs).log_prob(batch_act)
                batch_ratios = torch.exp(new_logp - old_logp)
                ratios.append(batch_ratios)
            ratios = torch.cat(ratios)
            new_pi_score = (ratios * adv).mean()
            if new_pi_score <= old_pi_score:
                return False

            return True

        alpha = 1.0
        for _ in range(self.ls_iters):
            ls_success = cond(alpha)
            if ls_success:
                break
            alpha *= self.ls_step

        if ls_success:
            new_theta = old_theta + alpha * s
            self.pi_params.flat = new_theta

        # Optimize value network
        for _ in range(self.value_optim_iters):
            self.v_optim.zero_grad(set_to_none=True)
            v_loss = nn_F.mse_loss(compute_v(), ret)
            v_loss.backward()
            self.v_optim.step()

        self.board.add_scalar("train/vpg_pi_loss", vpg_pi_loss, step=self.epoch_idx)
        self.board.add_scalar("train/V_loss", v_loss, step=self.epoch_idx)
