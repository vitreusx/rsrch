from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from rsrch import nn
from rsrch.exp import Experiment, timestamp
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.nn.utils import safe_mode
from rsrch.rl import data as data_rl
from rsrch.rl import gym
from rsrch.rl.data import rollout
from rsrch.rl.utils import polyak
from rsrch.utils import cron, data, repro
from rsrch.utils.early_stop import EarlyStopping

from . import ac, amp, config, env, wm


class Dataset(data.IterableDataset):
    """A dataset for DreamerV2. Slightly? changed from the original one, mostly
    because the original implementation is incomprehensible, and the paper
    doesn't really help. Anyways, here's how it works:
    - select B episodes, where B is batch size;
    - take slices of them, given by subseq_len parameter;
    - yield batches of consecutive slices of length slice_len from each of the
      B sequences.
    """

    KEYS = Literal["obs", "act", "reward", "term", "index", "seq_idx"]

    def __init__(
        self,
        env_f: env.Factory,
        eps: dict[int, dict],
        batch_size: int,
        slice_len: int,
        ongoing: bool = False,
        subseq_len: int | tuple[int, int] | None = None,
        prioritize_ends: bool = False,
        mode: Literal["train", "val"] = "train",
    ):
        super().__init__()
        self.env_f = env_f
        self.eps = eps
        self.batch_size = batch_size
        self.slice_len = slice_len
        self.ongoing = ongoing
        self.prioritize_ends = prioritize_ends
        self.mode = mode

        if subseq_len is None:
            self.minlen, self.maxlen = 1, None
        elif isinstance(subseq_len, tuple):
            self.minlen, self.maxlen = subseq_len
        else:
            self.minlen, self.maxlen = subseq_len, subseq_len
        self.minlen = max(self.minlen, self.slice_len)

    def __iter__(self):
        if self.mode == "train":
            yield from self._train_iter()
        else:
            yield from self._val_iter()

    def _train_iter(self):
        cur_eps: dict[int, dict[self.KEYS, list]] = {}
        free_idx = 0

        while True:
            ep_ids = [*self.eps]

            for ep_idx in [*cur_eps]:
                seq, begin = cur_eps[ep_idx]
                end = begin + self.slice_len
                if end > len(seq["obs"]):
                    del cur_eps[ep_idx]
                    continue

            while len(cur_eps) < self.batch_size:
                ep_id = np.random.choice(ep_ids)
                seq = {**vars(self.eps[ep_id])}
                del seq["info"]
                if not self.ongoing and not seq["term"]:
                    continue

                total = len(seq["obs"])
                if total < self.minlen:
                    continue

                length = total
                if self.maxlen:
                    length = min(length, self.maxlen)
                length -= np.random.randint(self.minlen)
                length = max(self.minlen, length)

                upper = total - length + 1
                if self.prioritize_ends:
                    upper += self.minlen

                index = min(np.random.randint(upper), total - length)

                subseq = self._subseq(seq, index, index + length)
                subseq["seq_idx"] = free_idx
                cur_eps[free_idx] = [subseq, 0]

                free_idx += 1

            batch = []
            for ep_idx in [*cur_eps]:
                seq, begin = cur_eps[ep_idx]
                end = begin + self.slice_len

                subseq = {}
                for k, v in seq.items():
                    subseq[k] = v[begin:end] if k != "seq_idx" else v

                batch.append(subseq)
                cur_eps[ep_idx] = [seq, end]

            batch = self.collate_fn(batch)
            yield batch

    def _val_iter(self):
        batch = []
        for ep_id in self.eps:
            seq = {**vars(self.eps[ep_id])}

            total = len(seq["obs"])
            if total < self.minlen:
                continue

            seq = self._subseq(seq, 0, total)
            seq["seq_idx"] = ep_id

            for index in range(0, total, self.slice_len):
                end = min(index + self.slice_len, total)
                index = end - self.slice_len

                subseq = {}
                for k, v in seq.items():
                    subseq[k] = v[index:end] if k != "seq_idx" else v

                batch.append(subseq)
                if len(batch) == self.batch_size:
                    batch = self.collate_fn(batch)
                    yield batch
                    batch = []

    def _subseq(self, seq, begin, end):
        subseq = {}

        subseq["obs"] = seq["obs"][begin:end]

        subseq["act"] = seq["act"][max(begin - 1, 0) : end - 1]
        subseq["reward"] = seq["reward"][max(begin - 1, 0) : end - 1]
        if begin == 0:
            subseq["act"] = [np.zeros_like(seq["act"][0]), *seq["act"]]
            subseq["reward"] = [0.0, *seq["reward"]]

        subseq["index"] = np.arange(begin, end)

        subseq["term"] = np.zeros([end - begin], dtype=bool)
        total = len(seq["obs"])
        subseq["term"][-1] = seq["term"] and end == total

        return subseq

    def collate_fn(self, batch: list[dict[KEYS, list]]):
        out = {}

        out["obs"] = np.stack([x["obs"] for x in batch], axis=1)
        out["obs"] = self.env_f.move_obs(out["obs"])

        out["act"] = np.stack([x["act"] for x in batch], axis=1)
        out["act"] = self.env_f.move_act(out["act"], to="net")

        out["reward"] = np.stack([x["reward"] for x in batch], axis=1)
        device, dtype = out["obs"].device, out["obs"].dtype
        out["reward"] = torch.as_tensor(out["reward"], device=device, dtype=dtype)

        out["term"] = np.stack([x["term"] for x in batch], axis=1)
        out["term"] = torch.as_tensor(out["term"], device=device, dtype=torch.bool)

        out["index"] = np.stack([x["index"] for x in batch], axis=1)
        out["seq_idx"] = np.array([x["seq_idx"] for x in batch])

        return out


class TrainerBase(nn.Module):
    def __init__(self, cfg: config.Config):
        super().__init__()
        self.cfg = cfg
        self._setup_core()
        self._setup_debug()
        self._setup_env()
        self._setup_nets()
        self._setup_agent()

        self.buf: data_rl.Buffer

    def _setup_core(self):
        repro.fix_seeds(
            seed=self.cfg.seed,
            deterministic=self.cfg.debug.deterministic,
        )

        self.device = torch.device(self.cfg.device)
        amp.set_policy(compute_dtype=getattr(torch, self.cfg.dtype))

        self.env_step = self.agent_step = self.opt_step = 0
        self.pbar = tqdm(desc="DreamerV2", total=self.cfg.total.n)

        self.exp = Experiment(
            project="dreamerv2",
            run=f"{self.cfg.env.id}__{timestamp()}",
            config=asdict(self.cfg),
        )
        self.exp.boards += [Tensorboard(self.exp.dir / "board")]
        self.exp.register_step("env_step", lambda: self.env_step, default=True)

    def _setup_debug(self):
        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if self.cfg.debug.record_memory:
            torch.cuda.memory._record_memory_history()

        if self.cfg.debug.profile:
            self.prof = Profiler(
                device=self.device,
                traces_dir=self.exp.dir / "traces",
                schedule=dict(wait=5, warmup=1, active=3, repeat=4),
            )

    def _setup_env(self):
        self.env_f = env.make_factory(
            cfg=self.cfg.env,
            device=self.device,
            seed=self.cfg.seed,
        )
        self._frame_skip = getattr(self.env_f, "frame_skip", 1)
        self.train_envs = self.env_f.vector_env(1, mode="train")

    def _setup_nets(self):
        self.wm = wm.WorldModel(
            obs_space=self.env_f.obs_space,
            act_space=self.env_f.act_space,
            cfg=self.cfg.wm,
        )

        self.ac = ac.ActorCritic(
            wm=self.wm,
            cfg=self.cfg.ac,
        )

        self.to(self.device)
        self.apply(self._tf_init)

    def _tf_init(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _setup_agent(self):
        self.train_agent = ac.Agent(
            env_f=self.env_f,
            ac=self.ac,
            cfg=self.cfg.agent,
            mode="prefill",
        )
        self.env_iter = iter(rollout.steps(self.train_envs, self.train_agent))

        self._ep_ids = defaultdict(lambda: None)
        self._ep_rets = defaultdict(lambda: 0.0)

    def _make_until(self, cfg: config.Until):
        return cron.Until(
            step_fn=lambda: getattr(self, cfg.of),
            max_value=cfg.n,
        )

    def _make_every(self, cfg: config.Every, **kwargs):
        return cron.Every(
            **{
                "step_fn": lambda: getattr(self, cfg.of),
                "every": cfg.n,
                "iters": cfg.iters,
                **kwargs,
            }
        )

    def train(self):
        raise NotImplementedError()

    def take_env_step(self):
        env_idx, step = next(self.env_iter)
        self._ep_ids[env_idx] = self.buf.push(self._ep_ids[env_idx], step)
        self._ep_rets[env_idx] += step.reward
        if step.done:
            self.exp.add_scalar("train/ep_ret", self._ep_rets[env_idx])
            del self._ep_ids[env_idx]
            del self._ep_rets[env_idx]

        self.env_step += self._frame_skip
        self.agent_step += 1
        diff = getattr(self, self.cfg.total.of) - self.pbar.n
        self.pbar.update(diff)


class BasicTrainer(TrainerBase):
    def __init__(self, cfg: config.Config):
        assert cfg.trainer.mode == "basic"
        super().__init__(cfg)

        self._setup_data()
        self._env_thr = ThreadPoolExecutor(max_workers=1)

    def _setup_data(self):
        cfg = vars(self.cfg.dataset)

        self.buf = self.env_f.buffer(cfg["capacity"])
        del cfg["capacity"]

        self.eps = data_rl.EpisodeView(self.buf)
        self.dataset = Dataset(env_f=self.env_f, eps=self.eps, **cfg)

        self.batch_iter = iter(self.dataset)
        self._states = {}

    def run(self):
        do_prefill = self._make_until(self.cfg.prefill)

        self.train_agent.mode = "prefill"
        while do_prefill:
            self.take_env_step()
        self.train_agent.mode = "train"

        self.should_run = self._make_until(self.cfg.total)
        self.should_log = self._make_every(self.cfg.log_every, iters=None)

        cfg = self.cfg.trainer.basic
        self.should_opt = self._make_every(cfg.sched)

        while self.should_run:
            self._train_step()

    def _train_step(self):
        # Idea: interleave env interaction and optimization by squeezing the
        # former during the periods when torch releases GIL in backward (hence
        # the thread pool.)
        batch = next(self.batch_iter)
        task = self._env_thr.submit(self._take_steps)
        states, term = self._train_wm(batch)
        self._train_ac(states, term)
        task.result()
        self.opt_step += 1
        self.eps.update()

    def _take_steps(self):
        while not self.should_opt:
            self.take_env_step()

    def _train_wm(self, batch):
        start_pos = []
        for idx in range(batch["obs"].shape[1]):
            seq_idx = batch["seq_idx"][0]
            index = batch["index"][0, idx] - 1
            start_pos.append((seq_idx, index))

        for pos in [*self._states]:
            if pos not in start_pos:
                del self._states[pos]

        start, is_first = [], []
        for pos in start_pos:
            is_first.append(pos not in self._states)
            if is_first[-1]:
                start_ = self.wm.rssm.initial(self.device)
            else:
                start_ = self._states[pos]
            start.append(start_)

        start = torch.stack(start)
        is_first = torch.tensor(is_first, device=self.device)

        mets, states = self.wm.train_step(batch, start, is_first)

        final = states[-1]
        for idx, state in enumerate(final):
            seq_idx, index = start_pos[idx]
            index += len(batch["obs"])
            self._states[seq_idx, index] = state

        if self.should_log:
            for k, v in mets.items():
                self.exp.add_scalar(f"wm/{k}", v)

        return states.flatten(), batch["term"].flatten()

    def _train_ac(self, states, term):
        mets = self.ac.train_step(states, term)

        if self.should_log:
            for k, v in mets.items():
                self.exp.add_scalar(f"ac/{k}", v)


class IterativeTrainer(TrainerBase):
    def __init__(self, cfg: config.Config):
        assert cfg.trainer.mode == "iterative"
        super().__init__(cfg)

        self._setup_data()
        self._env_thr = ThreadPoolExecutor(max_workers=1)
        self._wm_opt_step, self._ac_opt_step = 0, 0

    def _setup_data(self):
        cfg = vars(self.cfg.dataset)
        self.buf = self.env_f.buffer(cfg["capacity"])
        del cfg["capacity"]
        self._dataset_cfg = cfg

        self.eps = data_rl.EpisodeView(self.buf)
        self.dataset = Dataset(env_f=self.env_f, eps=self.eps, **cfg)

        self._states = {}
        self.batch_iter = iter(self.dataset)

    def run(self):
        do_prefill = self._make_until(self.cfg.prefill)

        self.train_agent.mode = "prefill"
        while do_prefill:
            self.take_env_step()
        self.train_agent.mode = "train"

        self.should_run = self._make_until(self.cfg.total)
        self.should_log = self._make_every(self.cfg.log_every, iters=None)

        cfg = self.cfg.trainer.iterative
        self.should_train_ac = self._make_every(cfg.ac_sched)
        self.should_train_wm = self._make_every(cfg.wm_sched)

        while self.should_run:
            # We wish to interleave env interaction and opt steps, if possible.
            # The way we do it is as follows: figure out how many env steps to
            # take until we do an optimization step, schedule them in another
            # thread, and run the opt steps. The env thread's going to run when
            # torch releases GIL in the backward call.

            num_steps = 0
            while True:
                train_wm_ = bool(self.should_train_wm)
                train_ac_ = bool(self.should_train_ac)
                if train_wm_ or train_ac_:
                    break

                self.env_step += self._frame_skip
                self.agent_step += 1
                num_steps += 1

            self.env_step -= num_steps * self._frame_skip
            self.agent_step -= num_steps

            if not (cfg.enable_for_wm and cfg.enable_for_ac):
                batch = next(self.batch_iter)

            task = self._env_thr.submit(self._take_steps, num_steps)

            ac_batch = None
            if train_wm_:
                if cfg.enable_for_wm:
                    self._train_wm_iter()
                else:
                    ac_batch = self._train_wm_step(batch)

            if train_ac_:
                if cfg.enable_for_ac:
                    self._train_ac_iter()
                else:
                    if ac_batch is None:
                        with torch.no_grad():
                            ac_batch = self._get_ac_batch(batch)
                    self._train_ac_step(ac_batch)

            task.result()
            self.eps.update()

    def _take_steps(self, num_steps: int):
        for _ in range(num_steps):
            self.take_env_step()

    def _train_wm_iter(self):
        cfg = self.cfg.trainer.iterative
        has_converged = EarlyStopping(**vars(cfg.stop_criteria))

        # NOTE: torch.rand(n)[:m] = torch.rand(m). Because of this, subsequent
        # training iterations will never assign episodes previously used for
        # training into the validation set.

        g = np.random.default_rng(seed=self.cfg.seed + 3)

        slice_ = slice(self.eps.ids.start, self.eps.ids.stop)
        is_val_ep = g.random(slice_.stop)[slice_] < 0.15
        if not is_val_ep.any():
            is_val_ep[0] = True

        val_ids = {*np.where(is_val_ep)[0]}
        train_eps, val_eps = {}, {}
        for ep_id, ep in self.eps.items():
            if ep_id in val_ids:
                val_eps[ep_id] = ep
            else:
                train_eps[ep_id] = ep

        train_ds = Dataset(self.env_f, train_eps, **self._dataset_cfg, mode="train")
        batch_iter = iter(train_ds)
        val_ds = Dataset(self.env_f, val_eps, **self._dataset_cfg, mode="val")

        pbar = self.exp.pbar(desc="(WM train iter)", initial=self._wm_opt_step)

        while True:
            if self._wm_opt_step % cfg.val_every == 0:
                val_loss, val_n = 0.0, 0

                self._states.clear()
                for batch in val_ds:
                    start, is_first, start_pos = self._get_start_states(batch)
                    batch_loss, final = self.wm.val_step(batch, start, is_first)
                    self._update_start_states(final, start_pos, val_ds.slice_len)
                    val_loss += batch_loss
                    val_n += 1
                self._states.clear()

                val_loss /= val_n
                self.exp.add_scalar("wm/val_loss", val_loss, step=self._wm_opt_step)
                if has_converged(val_loss, self._wm_opt_step):
                    break

            batch = next(batch_iter)
            self._train_wm_step(batch, step=self._wm_opt_step)
            self._wm_opt_step += 1
            self.opt_step += 1
            pbar.update()

    def _train_wm_step(self, batch, step=None):
        start, is_first, pos = self._get_start_states(batch)
        mets, states = self.wm.train_step(batch, start, is_first)

        self._update_start_states(states[-1], pos, self.dataset.slice_len)

        if self.should_log:
            for k, v in mets.items():
                self.exp.add_scalar(f"wm/{k}", v, step=step)

        return states.flatten(), batch["term"].flatten()

    def _get_start_states(self, batch):
        batch_size = batch["obs"].shape[1]

        start_pos = []
        for idx in range(batch_size):
            seq_idx = batch["seq_idx"][0]
            index = batch["index"][0, idx] - 1
            start_pos.append((seq_idx, index))

        start, is_first = [], []
        for pos in start_pos:
            is_first.append(pos not in self._states)
            if is_first[-1]:
                start_ = self.wm.rssm.initial(self.device)
            else:
                start_ = self._states[pos]
            start.append(start_)

        start = torch.stack(start)
        is_first = torch.tensor(is_first, device=self.device)
        return start, is_first, start_pos

    def _update_start_states(self, final, start_pos, slice_len):
        self._states = {}
        for idx, state in enumerate(final):
            seq_idx, index = start_pos[idx]
            index += slice_len
            self._states[seq_idx, index] = state

    def _train_ac_iter(self):
        ...

    def _get_ac_batch(self, batch):
        start, is_first = self._get_start_states(batch)[:2]
        states = self.wm.get_states(batch, start, is_first)
        return states.flatten(), batch["term"].flatten()

    def _train_ac_step(self, batch, step=None):
        mets = self.ac.train_step(*batch)

        if self.should_log:
            for k, v in mets.items():
                self.exp.add_scalar(f"ac/{k}", v, step=step)


def main():
    cfg = config.load(Path(__file__).parent / "config.yml")

    presets = config.load(Path(__file__).parent / "presets.yml")
    preset_names = ["atari", "debug"]
    # preset_names = ["atari", "debug", "atari_der"]
    for preset in preset_names:
        config.add_preset_(cfg, presets, preset)

    cfg = config.parse(cfg, config.Config)

    if cfg.trainer.mode == "basic":
        trainer = BasicTrainer(cfg)
    elif cfg.trainer.mode == "iterative":
        trainer = IterativeTrainer(cfg)
    else:
        raise NotImplementedError(cfg.trainer.mode)

    trainer.run()


if __name__ == "__main__":
    main()
