from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from rsrch.exp import Experiment, timestamp
from rsrch.exp.board.tensorboard import Tensorboard
from rsrch.exp.profile import Profiler
from rsrch.rl import data as data_rl
from rsrch.rl import gym
from rsrch.rl.data import rollout
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
        cur_eps = {}
        free_idx = 0

        while True:
            batch = {}
            while len(batch) < self.batch_size:
                while len(cur_eps) < self.batch_size:
                    ep_id = np.random.choice(self.eps.ids)
                    seq = self.eps[ep_id]
                    if not self.ongoing and not seq.term:
                        continue

                    total = len(seq.obs)
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
                    cur_eps[free_idx] = [seq, index, index + length]
                    free_idx += 1

                idxes = {*cur_eps} - {*batch}
                for ep_idx in idxes:
                    seq, start, stop = cur_eps[ep_idx]
                    cur_stop = start + self.slice_len
                    if cur_stop > stop:
                        del cur_eps[ep_idx]
                        continue

                    seq: data_rl.Seq

                    obs = seq.obs[start:cur_stop]
                    is_first = start == 0
                    if is_first:
                        act = seq.act[: cur_stop - 1]
                        act = [act[0], *act]
                        reward = seq.reward[: cur_stop - 1]
                        reward = [reward[0], *reward]
                    else:
                        act = seq.act[start - 1 : cur_stop - 1]
                        reward = seq.reward[start - 1 : cur_stop - 1]
                    term = seq.term

                    batch[ep_idx] = {
                        "seq": data_rl.Seq(obs, act, reward, term),
                        "is_first": is_first,
                    }
                    cur_eps[ep_idx] = [seq, start + length, stop]

            yield batch


class Runner:
    def main(self):
        cfg = config.load(Path(__file__).parent / "config.yml")

        presets = config.load(Path(__file__).parent / "presets.yml")
        preset_names = ["atari", "debug"]
        for preset in preset_names:
            config.add_preset_(cfg, presets, preset)

        self.cfg = config.parse(cfg, config.Config)

        repro.fix_seeds(seed=self.cfg.seed, deterministic=False)
        self._seedseq = np.random.SeedSequence(self.cfg.seed)

        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype)
        # torch.autograd.set_detect_anomaly(True)

        self.env_step = self.agent_step = 0
        self.pbar = tqdm(desc="DreamerV2", total=self.cfg.total.n)

        self.exp = Experiment(
            project="dreamerv2",
            run=f"{self.cfg.env.id}__{timestamp()}",
            config=cfg,
        )
        self.exp.boards += [Tensorboard(self.exp.dir / "board")]
        self.exp.register_step("env_step", lambda: self.env_step, default=True)

        self.prof = Profiler(
            device=self.device,
            traces_dir=self.exp.dir / "traces",
            schedule=dict(wait=5, warmup=1, active=3, repeat=4),
        )

        self._opt_step = self.prof.profiled(self._opt_step, "opt_step")

        self._setup_env()
        self._setup_data()

        self.wm = wm.WorldModel(
            obs_space=self.env_f.obs_space,
            act_space=self.env_f.act_space,
            cfg=self.cfg.wm,
            dtype=self.dtype,
        )

        self.ac = ac.ActorCritic(
            wm=self.wm,
            cfg=self.cfg.ac,
            dtype=self.dtype,
        )

        self.wm = self.wm.to(self.device)
        self.ac = self.ac.to(self.device)

        self._setup_agent()

        self._train()

    def _setup_env(self):
        self.env_f = env.make_factory(
            cfg=self.cfg.env,
            device=self.device,
            seed=self.cfg.seed,
        )
        self._frame_skip = getattr(self.env_step, "frame_skip", 1)
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
        should_opt = self._make_every(self.cfg.train_every)

        self.train_agent.mode = "train"
        while do_train:
            if should_eval:
                self._eval_epoch()
            self._take_step()
            if should_opt:
                self._opt_step()

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

    def _opt_step(self):
        batch = next(self.batch_iter)

        seq_batch = self.env_f.collate_fn([x["seq"] for x in batch.values()])
        is_first = np.array([x["is_first"] for x in batch.values()])
        is_first = torch.asarray(is_first, device=self.device)

        self._states = {k: v for k, v in self._states.items() if k in batch}

        for k in batch:
            if k not in self._states:
                self._states[k] = self.wm.rssm.initial(self.device, self.dtype)

        state = torch.stack([self._states[k] for k in batch])

        mets, last_state, is_term = self.wm.opt_step(seq_batch, state, is_first)

        for k, s in zip(batch, last_state):
            self._states[k] = s

        for k, v in mets.items():
            self.exp.add_scalar(f"wm/{k}", v)

        mets = self.ac.opt_step(last_state, is_term)

        for k, v in mets.items():
            self.exp.add_scalar(f"ac/{k}", v)


if __name__ == "__main__":
    Runner().main()
