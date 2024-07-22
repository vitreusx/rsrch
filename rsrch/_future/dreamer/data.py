import numpy as np
from torch.utils import data

from rsrch._future import rl


class Dataset(data.IterableDataset):
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
                seq, begin = cur_eps[ep_idx]
                end = begin + self.slice_len
                if end > len(seq["obs"]):
                    del cur_eps[ep_idx]
                    continue

            while len(cur_eps) < self.batch_size:
                ep_id = next(ep_id_iter)
                seq = self.buf[ep_id]
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
                subseq["seq_idx"] = next_idx
                cur_eps[next_idx] = [subseq, 0]

                next_idx += 1

            batch = []
            for ep_idx in [*cur_eps]:
                seq, begin = cur_eps[ep_idx]
                end = begin + self.slice_len

                subseq = {}
                for k, v in seq.items():
                    if hasattr(v, "__getitem__"):
                        subseq[k] = v[begin:end]
                    else:
                        subseq[k] = v

                batch.append(subseq)
                cur_eps[ep_idx] = [seq, end]

            yield from batch

    def dataloader(self):
        return data.DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def _subseq(self, seq: dict, start, stop):
        r = {}
        for k, vs in seq.items():
            if k == "act":
                r["act"] = vs[max(start - 1, 0) : stop - 1]
                if start == 0:
                    r["act"] = [r["act"][0], *r["act"]]
            elif hasattr(vs, "__getitem__"):
                r[k] = vs[start:stop]
            else:
                r[k] = vs
        return r
