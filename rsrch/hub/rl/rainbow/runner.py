import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import islice
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import torch
from torch import Tensor, nn

from rsrch import exp as exp
from rsrch import rl, spaces
from rsrch.nn import noisy
from rsrch.nn.utils import safe_mode
from rsrch.rl import data, gym
from rsrch.rl.utils import polyak
from rsrch.utils import cron, repro
from rsrch.utils import sched as sched

from . import config, nets
from .distq import ValueDist


class QAgent(gym.agents.Memoryless):
    def __init__(self, q: nets.Q, obs_space, act_space, val: bool = False):
        super().__init__(obs_space, act_space)
        self.q = q
        self.device = next(self.q.parameters()).device
        self.val = val

    @torch.inference_mode()
    def _policy(self, obs: Tensor):
        obs = obs.to(self.device)
        q = self.q(obs, val_mode=self.val)
        if isinstance(q, ValueDist):
            q = q.mean
        act = q.argmax(-1)
        return act.cpu()


class EpsVecAgent(gym.VecAgentWrapper):
    def __init__(self, agent: gym.VecAgent, eps: float):
        super().__init__(agent)
        self.eps = eps

    def policy(self, idxes):
        if np.random.rand() < self.eps:
            return self.act_space.sample((len(idxes),))
        else:
            return super().policy(idxes)


@dataclass
class SliceBatch:
    obs: Tensor
    act: Tensor
    reward: Tensor
    term: Tensor

    def to(self, device: torch.device):
        return SliceBatch(
            obs=self.obs.to(device),
            act=self.act.to(device),
            reward=self.reward.to(device),
            term=self.term.to(device),
        )


class Loader:
    def __init__(self, slices: rl.data.SliceView, batch_size: int):
        self.slices = slices
        self.batch_size = batch_size

    def __iter__(self):
        slice_iter = iter(self.slices)
        prioritized = isinstance(self.slices.sampler, data.PSampler)

        while True:
            batch, prio_seq, pos_seq = [], [], []
            for _ in range(self.batch_size):
                seq, pos = next(slice_iter)
                batch.append(seq)
                if prioritized:
                    pos_seq.append(pos)
                    prio_seq.append(self.slices.sampler[pos])

            batch = self.collate_fn(batch)
            if prioritized:
                prio_seq = torch.tensor(prio_seq)
                yield batch, prio_seq, pos_seq
            else:
                yield batch

    def collate_fn(self, batch: list[list[dict]]):
        batch_size, seq_len = len(batch), len(batch[0])

        obs, act, reward, term = [], [], [], []
        for t in range(seq_len):
            for idx in range(batch_size):
                step = batch[idx][t]
                obs.append(step["obs"])
                if t > 0:
                    act.append(step["act"])
                    reward.append(step["reward"])
                if t == seq_len - 1:
                    term.append(step["term"])

        obs, act = torch.stack(obs), torch.stack(act)
        reward = torch.as_tensor(np.array(reward))
        term = torch.as_tensor(np.array(term))

        obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
        act = act.reshape(seq_len - 1, batch_size, *act.shape[1:])
        reward = reward.reshape(seq_len - 1, batch_size)
        term = term.reshape(batch_size)

        return SliceBatch(obs, act, reward, term)


class Runner:
    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    def main(self):
        self.prepare()
        if self.cfg.mode == "train":
            self.train()
        elif self.cfg.mode == "sample":
            self.sample()

    def prepare(self):
        repro.seed_all(self.cfg.random.seed)

        self.sdk = rl.sdk.make(self.cfg.env)
        assert isinstance(self.sdk.obs_space, spaces.torch.Image)
        assert isinstance(self.sdk.act_space, spaces.torch.Discrete)

        self.exp = exp.Experiment(
            project="rainbow",
            prefix=self.sdk.id,
            config=asdict(self.cfg),
        )
        self.exp.boards.append(exp.board.Tensorboard(self.exp.dir))

        self.device = torch.device(self.cfg.device)

        # torch.autograd.set_detect_anomaly(True)

        self.env_step, self.opt_step, self.agent_step = 0, 0, 0
        self.exp.register_step("env_step", lambda: self.env_step, default=True)
        self.exp.register_step("opt_step", lambda: self.opt_step)
        self.exp.register_step("agent_step", lambda: self.agent_step)

        def make_qf():
            qf = nets.Q(self.cfg, self.sdk.obs_space, self.sdk.act_space)
            qf = qf.to(self.device)
            qf = qf.share_memory()
            return qf

        self.qf, self.qf_t = make_qf(), make_qf()
        self.qf_opt = self.cfg.opt.optimizer(self.qf.parameters())

        self.buf = rl.data.Buffer()
        self.buf = self.sdk.wrap_buffer(self.buf)

        if self.cfg.mode == "train":
            self._prepare_train()

        if self.cfg.resume is not None:
            self.load(self.cfg.resume)

    def _prepare_train(self):
        self.buf = rl.data.Observable(self.buf)

        if self.cfg.prioritized.enabled:
            self.sampler = data.PSampler()
        else:
            self.sampler = data.Sampler()

        self.slices = data.SliceView(self.cfg.data.slice_len, sampler=self.sampler)
        self.buf.attach(hook=self.slices, replay=True)

        self.buf = rl.data.SizeLimited(self.buf, self.cfg.data.capacity)

        self.train_loader = Loader(self.slices, self.cfg.opt.batch_size)
        self.train_iter = iter(self.train_loader)

        self.envs = self.sdk.make_envs(self.cfg.num_envs, mode="train")

        self.agent_eps = self._make_sched(self.cfg.expl.eps)
        self.expl_eps = self.agent_eps()
        self.agent = EpsVecAgent(
            QAgent(self.qf, self.sdk.obs_space, self.sdk.act_space, val=False),
            self.expl_eps,
        )

        self.env_iter = iter(self.sdk.rollout(self.envs, self.agent))
        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)
        self._env_steps = defaultdict(lambda: 0)

        self.val_envs = self.sdk.make_envs(self.cfg.val.envs, mode="val")
        self.val_agent = QAgent(
            self.qf, self.sdk.obs_space, self.sdk.act_space, val=True
        )

        amp_enabled = self.cfg.opt.dtype != "float32"
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.autocast = lambda: torch.autocast(
            device_type=self.device.type,
            dtype=getattr(torch, self.cfg.opt.dtype),
            enabled=amp_enabled,
        )

        gammas = torch.tensor(
            [self.cfg.gamma**i for i in range(self.cfg.data.slice_len - 1)]
        )
        self.gammas = gammas.to(self.device)
        self.final_gamma = self.cfg.gamma ** (self.cfg.data.slice_len - 1)

        self.prio_exp = self._make_sched(self.cfg.prioritized.prio_exp)
        self.is_coef_exp = self._make_sched(self.cfg.prioritized.is_coef_exp)

        self.should_update_q = self._make_every(self.cfg.nets.polyak)
        self.tau = self.cfg.nets.polyak["tau"]

    def _make_every(self, cfg: dict):
        if isinstance(cfg["every"], dict):
            every, unit = cfg["every"]["n"], cfg["every"]["of"]
        else:
            every, unit = cfg["every"], "env_step"

        return cron.Every(
            step_fn=lambda: getattr(self, unit),
            period=every,
            iters=cfg.get("iters", 1),
        )

    def _make_until(self, cfg):
        if isinstance(cfg, dict):
            max_value, unit = cfg["n"], cfg["of"]
        else:
            max_value, unit = cfg, "env_step"

        return cron.Until(
            step_fn=lambda: getattr(self, unit),
            max_value=max_value,
        )

    def _make_sched(self, cfg):
        if isinstance(cfg, dict):
            desc, unit = cfg["desc"], cfg["of"]
        else:
            desc, unit = cfg, "env_step"
        return sched.Auto(desc, lambda: getattr(self, unit))

    def _make_pbar(self, until: cron.Until, *args, **kwargs):
        return self.exp.pbar(
            *args,
            **kwargs,
            total=until.max_value,
            initial=until.step_fn(),
        )

    def train(self):
        self.should_log = self._make_every(self.cfg.log)

        should_warmup = self._make_until(self.cfg.warmup)
        with self._make_pbar(should_warmup, desc="Warmup") as self.pbar:
            while should_warmup:
                self.take_env_step()

        should_val = self._make_every(self.cfg.val.sched)
        should_save_ckpt = self._make_every(self.cfg.ckpts.sched)
        should_opt = self._make_every(self.cfg.opt.sched)

        should_train = self._make_until(self.cfg.total)
        with self._make_pbar(should_train, desc="Train") as self.pbar:
            while should_train:
                if should_val:
                    self._val_epoch()

                if should_save_ckpt:
                    self._save_ckpt()

                while should_opt:
                    self._opt_step()

                self.take_env_step()

    def _val_epoch(self):
        val_iter = self.exp.pbar(
            islice(self._val_ret_iter(), 0, self.cfg.val.episodes),
            desc="Val",
            total=self.cfg.val.episodes,
            leave=False,
        )
        val_rets = [*val_iter]

        self.exp.add_scalar("val/mean_ret", np.mean(val_rets))
        self._last_val_ret = np.mean(val_rets)

    def _opt_step(self):
        batch_data = next(self.train_iter)

        if self.cfg.prioritized.enabled:
            batch, is_coefs, pos = batch_data
            is_coefs = is_coefs.to(self.device)
        else:
            batch = batch_data
        batch = batch.to(self.device)

        if self.cfg.aug.rew_clip is not None:
            batch.reward.clamp_(*self.cfg.aug.rew_clip)

        with torch.no_grad():
            with self.autocast():
                next_q_eval = self.qf_t(batch.obs[-1])

                if isinstance(next_q_eval, ValueDist):
                    if self.cfg.double_dqn:
                        next_q_act: ValueDist = self.qf(batch.obs[-1])
                    else:
                        next_q_act = next_q_eval
                    act = next_q_act.mean.argmax(-1)
                    target = next_q_eval.gather(-1, act[..., None])
                    target = target.squeeze(-1)
                elif isinstance(next_q_eval, Tensor):
                    if self.cfg.double_dqn:
                        next_q_act: Tensor = self.qf(batch.obs[-1])
                        act = next_q_act.argmax(-1)
                        target = next_q_eval.gather(-1, act[..., None])
                        target = target.squeeze(-1)
                    else:
                        target = next_q_eval.max(-1).values

                gamma_t = (1.0 - batch.term.float()) * self.final_gamma
                returns = (batch.reward * self.gammas.unsqueeze(-1)).sum(0)
                target = returns + gamma_t * target

        with self.autocast():
            qv = self.qf(batch.obs[0])
            pred = qv.gather(-1, batch.act[0][..., None]).squeeze(-1)

            if isinstance(target, ValueDist):
                # prio = q_losses = ValueDist.proj_kl_div(target, pred)
                prio = q_losses = ValueDist.apx_w1_div(target, pred)
            else:
                prio = (pred - target).abs()
                q_losses = (pred - target).square()

            if self.cfg.prioritized.enabled:
                is_coefs = torch.as_tensor(is_coefs, device=self.device)
                q_losses = (is_coefs ** self.is_coef_exp()) * q_losses

            loss = q_losses.mean()

        self.qf_opt.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.qf_opt)
        if self.cfg.opt.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.qf.parameters(), self.cfg.opt.grad_clip)
        self.scaler.step(self.qf_opt)
        self.scaler.update()

        self.opt_step += 1

        if self.should_log:
            self.exp.add_scalar("train/loss", loss)
            if isinstance(pred, ValueDist):
                self.exp.add_scalar("train/mean_q_pred", pred.mean.mean())
            else:
                self.exp.add_scalar("train/mean_q_pred", pred.mean())

        if self.cfg.prioritized.enabled:
            prio_exp = self.prio_exp()
            prio = prio.float().detach().cpu().numpy() ** prio_exp
            for prio_, pos_ in zip(prio, pos):
                self.sampler[pos_] = prio_

        while self.should_update_q:
            polyak.update(self.qf, self.qf_t, self.tau)

    def take_env_step(self):
        env_idx, (step, final) = next(self.env_iter)

        self._ep_ids[env_idx] = self.buf.push(self._ep_ids[env_idx], step, final)

        self._ep_rets[env_idx] += step.get("reward", 0.0)
        if final:
            self.exp.add_scalar("train/ep_ret", self._ep_rets[env_idx])
            del self._ep_ids[env_idx]
            del self._ep_rets[env_idx]

        self.agent_step += 1

        # Some environments skip frames (each action is repeated, say, four times.) For the purposes of benchmarking and keeping the comparisons fair, we need to keep track of the "real" number of env steps. This is done by adding "total_steps" field for all steps produced by the SDKs.
        if "total_steps" in step:
            diff = step["total_steps"] - self._env_steps[env_idx]
            self.env_step += diff
            self._env_steps[env_idx] = step["total_steps"]
        else:
            diff = 1
            self.env_step += 1

        self.pbar.update(self.env_step - self.pbar.n)

    def _val_ret_iter(self):
        val_ep_rets = defaultdict(lambda: 0.0)
        val_iter = iter(self.sdk.rollout(self.val_envs, self.val_agent))
        for env_idx, (step, final) in val_iter:
            val_ep_rets[env_idx] += step.get("reward", 0.0)
            if final:
                yield val_ep_rets[env_idx]
                del val_ep_rets[env_idx]

    def _save_ckpt(self):
        val_ret = self._last_val_ret
        path = (
            self.exp.dir
            / "ckpts"
            / f"env_step={self.env_step}-val_ret={val_ret:.2f}.pth"
        )
        self.save(path)

    def save(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state_dict()
        with open(ckpt_path, "wb") as f:
            torch.save(state, f)

    def _state_dict(self):
        state = {}
        state["rng"] = repro.RandomState.save()
        for name in ("qf", "qf_t", "qf_opt"):
            state[name] = getattr(self, name).state_dict()
        if self.cfg.ckpts.save_buf:
            state["env_buf"] = self.env_buf
        return state

    def load(self, ckpt_path: str | Path):
        ckpt_path = Path(ckpt_path)
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, map_location="cpu")
        self._load_state_dict(state)

    def _load_state_dict(self, state: dict):
        repro.RandomState.load(state["rng"])
        for name in ("qf", "qf_t", "qf_opt"):
            getattr(self, name).load_state_dict(state[name])
        if self.cfg.ckpts.save_buf:
            self.env_buf = state["env_buf"]
