from typing import Literal

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from rsrch import spaces
from rsrch._future import rl


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

        if subseq_len is None:
            self.minlen, self.maxlen = 1, None
        elif isinstance(subseq_len, tuple):
            self.minlen, self.maxlen = subseq_len
        else:
            self.minlen, self.maxlen = subseq_len, subseq_len
        self.minlen = max(self.minlen, self.slice_len)

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
                ep_id = next(ep_id_iter)
                seq = self.buf[ep_id]
                if not self.ongoing and not seq[-1]["term"]:
                    continue

                total = len(seq)
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

                cur_eps[next_idx] = {
                    "seq": seq,
                    "start": index,
                    "stop": index + length,
                    "ep_id": ep_id,
                    "pos": (next_idx, index),
                }
                next_idx += 1

            batch = []
            for ep_idx in [*cur_eps]:
                ep = cur_eps[ep_idx]

                cur_stop = ep["start"] + self.slice_len
                subseq = self._subseq(ep["seq"], ep["start"], cur_stop)
                for k in ("ep_id", "pos"):
                    subseq[k] = ep[k]
                batch.append(subseq)

                ep["start"] = cur_stop

            yield from batch

    def _subseq(self, seq: list[dict], start, stop):
        seq = seq[start:stop]

        res = {
            "obs": torch.stack([step["obs"] for step in seq]),
            "reward": torch.tensor(np.array([step["reward"] for step in seq])),
            "term": torch.tensor(np.array([step["term"] for step in seq])),
        }

        res["act"] = [step.get("act", None) for step in seq]
        if res["act"][0] is None:
            res["act"][0] = torch.zeros_like(res["act"][-1])
        res["act"] = torch.stack(res["act"])

        return res

    @staticmethod
    def collate_fn(batch):
        return {
            "obs": torch.stack([seq["obs"] for seq in batch], 1),
            "act": torch.stack([seq["act"] for seq in batch], 1),
            "reward": torch.stack([seq["reward"] for seq in batch], 1),
            "term": torch.stack([seq["term"] for seq in batch], 1),
            "ep_id": np.asarray([seq["ep_id"] for seq in batch]),
            "pos": [seq["pos"] for seq in batch],
        }
