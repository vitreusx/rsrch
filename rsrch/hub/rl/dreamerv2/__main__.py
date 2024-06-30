from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal

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

from . import ac, config, env, wm


class Dataset(data.IterableDataset):
    """A dataset for DreamerV2. Slightly? changed from the original one, mostly
    because the original implementation is incomprehensible, and the paper
    doesn't really help. Anyways, here's how it works:
    - select B episodes, where B is batch size;
    - take slices of them, given by subseq_len parameter;
    - yield batches of consecutive slices of length slice_len from each of the
      B sequences.
    """

    KEYS = Literal["obs", "act", "reward", "term", "first"]

    def __init__(
        self,
        env_f: env.Factory,
        capacity: int,
        batch_size: int,
        slice_len: int,
        ongoing: bool = False,
        subseq_len: int | tuple[int, int] | None = None,
        prioritize_ends: bool = False,
    ):
        super().__init__()
        self.env_f = env_f
        self.buf = env_f.buffer(capacity)
        self.eps = data_rl.EpisodeView(self.buf)
        self.batch_size = batch_size
        self.slice_len = slice_len
        self.ongoing = ongoing
        self.prioritize_ends = prioritize_ends

        if subseq_len is None:
            self.minlen, self.maxlen = 1, None
        elif isinstance(subseq_len, tuple):
            self.minlen, self.maxlen = subseq_len
        else:
            self.minlen, self.maxlen = subseq_len, subseq_len
        self.minlen = max(self.minlen, self.slice_len)

    def push(self, ep_id: int | None, step: data_rl.Step) -> int | None:
        ret = self.buf.push(ep_id, step)
        self.eps.update()
        return ret

    def __iter__(self):
        cur_eps: dict[int, dict[self.KEYS, list]] = {}
        free_idx = 0

        while True:
            batch = {}
            while len(batch) < self.batch_size:
                for ep_idx in [*cur_eps]:
                    seq, begin = cur_eps[ep_idx]
                    end = begin + self.slice_len
                    if end > len(seq["obs"]):
                        del cur_eps[ep_idx]
                        continue

                while len(cur_eps) < self.batch_size:
                    ep_id = np.random.choice(self.eps.ids)
                    seq = vars(self.eps[ep_id])
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

                    seq["obs"] = seq["obs"][index : index + length]

                    seq["act"] = seq["act"][max(index - 1, 0) : index + length - 1]
                    seq["reward"] = seq["reward"][
                        max(index - 1, 0) : index + length - 1
                    ]
                    if index == 0:
                        seq["act"] = [np.zeros_like(seq["act"][0]), *seq["act"]]
                        seq["reward"] = [0.0, *seq["reward"]]

                    seq["first"] = np.zeros([length], dtype=bool)
                    seq["first"][0] = True

                    term_ = seq["term"]
                    seq["term"] = np.zeros([length], dtype=bool)
                    seq["term"][-1] = term_ and index + length == total

                    cur_eps[free_idx] = [seq, 0]
                    free_idx += 1

                for ep_idx in [*cur_eps]:
                    seq, begin = cur_eps[ep_idx]
                    end = begin + self.slice_len
                    batch[ep_idx] = {k: v[begin:end] for k, v in seq.items()}
                    cur_eps[ep_idx] = [seq, end]

            ids = [*batch]
            batch = self._collate_fn([*batch.values()])
            yield ids, batch

    def _collate_fn(self, batch: list[dict[KEYS, list]]):
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

        out["first"] = np.stack([x["first"] for x in batch], axis=1)
        out["first"] = torch.as_tensor(out["first"], device=device, dtype=torch.bool)

        return out


class Runner(nn.Module):
    def main(self):
        cfg = config.load(Path(__file__).parent / "config.yml")

        presets = config.load(Path(__file__).parent / "presets.yml")
        preset_names = ["atari", "debug"]
        for preset in preset_names:
            config.add_preset_(cfg, presets, preset)

        self.cfg = config.parse(cfg, config.Config)

        repro.fix_seeds(
            seed=self.cfg.seed,
            deterministic=self.cfg.debug.deterministic,
        )
        self._seedseq = np.random.SeedSequence(self.cfg.seed)

        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype)

        self.env_step = self.agent_step = self.opt_step = 0
        self.pbar = tqdm(desc="DreamerV2", total=self.cfg.total.n)

        self.exp = Experiment(
            project="dreamerv2",
            run=f"{self.cfg.env.id}__{timestamp()}",
            config=cfg,
        )
        self.exp.boards += [Tensorboard(self.exp.dir / "board")]
        self.exp.register_step("env_step", lambda: self.env_step, default=True)

        if self.cfg.debug.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        if self.cfg.debug.record_memory:
            torch.cuda.memory._record_memory_history()

        if self.cfg.debug.profile:
            prof = Profiler(
                device=self.device,
                traces_dir=self.exp.dir / "traces",
                schedule=dict(wait=5, warmup=1, active=3, repeat=4),
            )
            # self._opt_step = prof.profiled(self._opt_step, "opt_step")
            # self._take_step = prof.profiled(self._take_step, "take_step")
            self._train_step = prof.profiled(self._train_step, "train_step")

        self._setup_env()
        self._setup_data()

        self.wm = wm.WorldModel(
            obs_space=self.env_f.obs_space,
            act_space=self.env_f.act_space,
            cfg=self.cfg.wm,
            device=self.device,
            dtype=self.dtype,
        )

        self.ac = ac.ActorCritic(
            wm=self.wm,
            cfg=self.cfg.ac,
            device=self.device,
            dtype=self.dtype,
        )

        self.apply(self._tf_init)
        polyak.sync(self.ac.critic, self.ac.target_critic)

        self._setup_agent()

        self._env_workers = ThreadPoolExecutor(max_workers=1)

        self._train()

    def _tf_init(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _setup_env(self):
        self.env_f = env.make_factory(
            cfg=self.cfg.env,
            device=self.device,
            seed=self.cfg.seed,
        )
        self._frame_skip = getattr(self.env_f, "frame_skip", 1)
        self.train_envs = self.env_f.vector_env(1, mode="train")

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

    def _setup_data(self):
        self.dataset = Dataset(
            self.env_f,
            **vars(self.cfg.dataset),
        )
        self.batch_iter = iter(self.dataset)
        self._states = {}

    def _train(self):
        do_prefill = self._make_until(self.cfg.prefill)

        self.train_agent.mode = "prefill"
        while do_prefill:
            self._take_step()

        do_train = self._make_until(self.cfg.total)
        should_eval = self._make_every(self.cfg.eval_every)
        self.should_opt = self._make_every(self.cfg.train_every)
        self.should_log = self._make_every(self.cfg.log_every)

        self.train_agent.mode = "train"
        while do_train:
            if should_eval:
                self._eval_epoch()
            self._train_step()

    def _train_step(self):
        # Idea: interleave env interaction and optimization by squeezing the
        # former during the periods when torch releases GIL in backward (hence
        # the thread pool.)
        batch = next(self.batch_iter)
        task = self._env_workers.submit(self._take_steps)
        self._opt_step(batch)
        task.result()

    def _take_steps(self):
        with safe_mode(self):
            while not self.should_opt:
                self._take_step()

    def _make_until(self, cfg: config.Until):
        return cron.Until(lambda: getattr(self, cfg.of), cfg.n)

    def _make_every(self, cfg: config.Every):
        return cron.Every(lambda: getattr(self, cfg.of), every=cfg.n, iters=cfg.iters)

    def _take_step(self):
        env_idx, step = next(self.env_iter)
        self._ep_ids[env_idx] = self.dataset.push(self._ep_ids[env_idx], step)
        self._ep_rets[env_idx] += step.reward
        if step.done:
            self.exp.add_scalar("train/ep_ret", self._ep_rets[env_idx])
            del self._ep_ids[env_idx]
            del self._ep_rets[env_idx]

        self.env_step += self._frame_skip
        self.agent_step += 1
        diff = getattr(self, self.cfg.total.of) - self.pbar.n
        self.pbar.update(diff)

    def _eval_epoch(self):
        ...

    def _opt_step(self, batch):
        ids, batch = batch

        for k in [*self._states]:
            if k not in ids:
                del self._states[k]

        for k in ids:
            if k not in self._states:
                self._states[k] = self.wm.rssm.initial(self.device, self.dtype)

        initial = torch.stack([self._states[k] for k in ids])
        wm_mets, states = self.wm.opt_step(batch, initial)

        for k, s in zip(ids, states[-1]):
            self._states[k] = s

        ac_mets = self.ac.opt_step(states.flatten(), batch["term"].flatten())

        if self.should_log:
            for k, v in wm_mets.items():
                self.exp.add_scalar(f"wm/{k}", v)
            for k, v in ac_mets.items():
                self.exp.add_scalar(f"ac/{k}", v)

        self.opt_step += 1


if __name__ == "__main__":
    Runner().main()
