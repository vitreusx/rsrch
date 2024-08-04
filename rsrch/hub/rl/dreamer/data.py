from collections import deque
from typing import Literal

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from rsrch import spaces
from rsrch._future import rl


class StateCache:
    def __init__(self, size: int):
        self.size = size
        self.states = {}
        self._order = deque(maxlen=size)

    def get(self, key):
        return self.states.get(key)

    def __setitem__(self, key, value):
        if key not in self.states:
            if len(self._order) == self.size:
                oldest = self._order.popleft()
                del self.states[oldest]
            self._order.append(key)
        self.states[key] = value

    def clear(self):
        self.states.clear()
        self._order.clear()


class Slices(data.IterableDataset):
    def __init__(
        self,
        buf: rl.Buffer,
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

        self.states = StateCache(self.batch_size)

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
                if not self.ongoing and not seq[-1]["term"]:
                    continue

                total = len(seq)

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
                subseq = self._subseq(ep["seq"], start, end)

                subseq["ep_id"] = ep["ep_id"]
                subseq["pos"] = (ep["seq_id"], start)
                subseq["start"] = self.states.get(subseq["pos"])

                batch.append(subseq)

                ep["start"] = end

            if len(batch) == 0:
                break

            yield from batch

    def _subseq(self, seq: list[dict], start, end):
        seq = seq[start:end]

        res = {
            "obs": torch.stack([step["obs"] for step in seq]),
            "reward": torch.tensor(np.array([step["reward"] for step in seq])),
            "term": torch.tensor(np.array([step["term"] for step in seq])),
        }

        res["act"] = [step.get("act") for step in seq]
        if res["act"][0] is None:
            res["act"][0] = torch.zeros_like(res["act"][-1])
        res["act"] = torch.stack(res["act"])

        return res

    def update_states(self, batch: dict, final):
        pos = batch["pos"]
        for (seq_id, offset), state in zip(pos, final):
            end = offset + self.slice_len
            self.states[seq_id, end] = state

    @staticmethod
    def collate_fn(batch):
        return {
            "obs": torch.stack([seq["obs"] for seq in batch], 1),
            "act": torch.stack([seq["act"] for seq in batch], 1),
            "reward": torch.stack([seq["reward"] for seq in batch], 1),
            "term": torch.stack([seq["term"] for seq in batch], 1),
            "ep_id": np.asarray([seq["ep_id"] for seq in batch]),
            "pos": [seq["pos"] for seq in batch],
            "start": [seq["start"] for seq in batch],
        }
