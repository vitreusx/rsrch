import threading
from collections import deque, namedtuple
from dataclasses import dataclass
from queue import Queue
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader

from rsrch import rl, spaces
from rsrch.nn.utils import over_seq

from .actor import Actor
from .common.types import Slices
from .common.utils import autocast
from .wm import WorldModel


class StateMap:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data = {}
        self._order = deque(maxlen=max_size)

    def get(self, key):
        return self._data.get(key)

    def __setitem__(self, key, value):
        if isinstance(key, list):
            for key_, value_ in zip(key, value):
                self[key_] = value_
        else:
            if key not in self._data:
                if len(self._order) == self.max_size:
                    oldest = self._order.popleft()
                    del self._data[oldest]
                self._order.append(key)
            self._data[key] = value

    def clear(self):
        self._data.clear()
        self._order.clear()


@dataclass
class BatchWM:
    seq: Slices
    h_0: list[Tensor | None]
    end_pos: list[tuple[int, int]]

    def to(self, device: torch.device):
        return BatchWM(
            seq=self.seq.to(device),
            h_0=[x.to(device) if x is not None else x for x in self.h_0],
            end_pos=self.end_pos,
        )


class RealLoaderWM(data.IterableDataset):
    def __init__(
        self,
        buf: rl.data.Buffer,
        sampler: data.Sampler,
        batch_size: int,
        slice_len: int,
        ongoing: bool = False,
        subseq_len: int | tuple[int, int] | None = None,
        prioritize_ends: bool = False,
    ):
        super().__init__()
        self.buf = buf
        self.sampler = sampler
        self.batch_size = batch_size
        self.slice_len = slice_len
        self.ongoing = ongoing
        self.subseq_len = subseq_len
        self.prioritize_ends = prioritize_ends

        if isinstance(subseq_len, tuple):
            self.minlen, self.maxlen = subseq_len
        elif subseq_len is not None:
            self.minlen, self.maxlen = subseq_len, subseq_len

        if hasattr(self, "minlen"):
            self.minlen = max(self.minlen, self.slice_len)

        self.h_0s = StateMap(self.batch_size)

    def __iter__(self):
        cur_eps = {}
        next_idx = 0
        ep_id_iter = iter(self.sampler)

        while True:
            for ep_idx in [*cur_eps]:
                ep = cur_eps[ep_idx]
                if ep["start"] + self.slice_len > ep["stop"]:
                    del cur_eps[ep_idx]
                    continue

            while len(cur_eps) < self.batch_size:
                ep_id = next(ep_id_iter, None)
                if ep_id is None:
                    break

                seq = self.buf[ep_id]

                total = len(seq)
                if total < self.slice_len:
                    continue

                if not self.ongoing and not (seq[-1]["term"] or seq[-1]["trunc"]):
                    continue

                if self.subseq_len is None:
                    index, length = 0, total
                else:
                    length = np.random.randint(self.minlen, self.maxlen + 1)
                    if self.prioritize_ends:
                        index = np.random.randint(total)
                        index = min(index, total - length)
                    else:
                        index = np.random.randint(total - length + 1)

                cur_eps[next_idx] = {
                    "seq": seq,
                    "start": index,
                    "stop": index + length,
                    "ep_id": ep_id,
                    "seq_id": next_idx,
                }
                next_idx += 1

            batch = []
            for ep_idx in [*cur_eps]:
                ep = cur_eps[ep_idx]

                end = min(ep["start"] + self.slice_len, ep["stop"])
                start = end - self.slice_len

                item = {}
                item["seq"] = self._subseq(ep["seq"], start, end)
                item["h_0"] = self.h_0s.get((ep["seq_id"], start))
                item["end_pos"] = (ep["seq_id"], end)

                batch.append(item)

                ep["start"] = end

            if len(batch) == 0:
                break

            yield self._collate_fn(batch)

    def _subseq(self, seq: list[dict], start, end):
        seq = seq[start:end]

        res = {
            "obs": torch.stack([step["obs"] for step in seq]),
            "reward": torch.tensor(np.array([step.get("reward", 0.0) for step in seq])),
            "term": torch.tensor(np.array([step.get("term", False) for step in seq])),
        }

        res["act"] = [step.get("act") for step in seq]
        if res["act"][0] is None:
            res["act"][0] = torch.zeros_like(res["act"][-1])
        res["act"] = torch.stack(res["act"])

        return Slices(**res)

    def _collate_fn(self, batch):
        return BatchWM(
            seq=torch.stack([item["seq"] for item in batch], dim=1),
            **{k: [item[k] for item in batch] for k in ("h_0", "end_pos")},
        )


def make_async(iterator: Iterator):
    """Make an iterator "asynchronous."

    To be precise, the fetching from the iterator is done by a separate thread."""

    batches = Queue(maxsize=1)

    def loader_fn():
        while True:
            batches.put(next(iterator))

    thr = threading.Thread(target=loader_fn, daemon=True)
    thr.start()

    while True:
        yield batches.get()


class DreamLoaderRL(data.IterableDataset):
    def __init__(
        self,
        real_slices: RealLoaderWM,
        wm: WorldModel,
        actor: Actor,
        batch_size: int,
        slice_len: int,
        device: torch.device | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.real_slices = real_slices
        self.wm = wm
        self.actor = actor
        self.batch_size = batch_size
        self.slice_len = slice_len
        self.device = device
        self.compute_dtype = compute_dtype

    def dream_from(self, h_0: Tensor, term: Tensor):
        self.wm.requires_grad_(False)

        with autocast(self.device, self.compute_dtype):
            states, actions = [h_0], []
            for _ in range(self.slice_len):
                policy = self.actor(states[-1].detach())
                enc_act = policy.rsample()
                actions.append(enc_act)
                next_state = self.wm.img_step(states[-1], enc_act).rsample()
                states.append(next_state)

            states = torch.stack(states)
            actions = torch.stack(actions)

            reward_dist = over_seq(self.wm.reward_dec)(states)
            reward = reward_dist.mode

            term_dist = over_seq(self.wm.term_dec)(states)
            term_ = term_dist.mean.contiguous()
            term_[0] = term.float()

        self.wm.requires_grad_(True)

        return Slices(states, actions, reward, term_)

    def __iter__(self):
        real_iter = iter(self.real_slices)

        while True:
            with torch.no_grad():
                with autocast(self.device, self.compute_dtype):
                    chunks, remaining = [], self.batch_size
                    while remaining > 0:
                        real_batch = next(real_iter)

                        out, _ = self.wm.observe(
                            input=(real_batch.seq.obs, real_batch.seq.act),
                            h_0=real_batch.h_0,
                        )
                        h_0 = out.states.flatten()
                        term = real_batch.seq.term.flatten()

                        if len(term) > remaining:
                            h_0, term = h_0[:remaining], term[:remaining]
                            remaining = 0
                        else:
                            remaining -= len(term)
                        chunks.append((h_0, term))

                    h_0 = torch.cat([h_0 for h_0, term in chunks], 0)
                    term = torch.cat([term for h_0, term in chunks], 0)

            yield self.dream_from(h_0, term)
