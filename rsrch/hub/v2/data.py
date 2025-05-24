from collections import defaultdict
from typing import Callable, Literal, Sequence, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Sampler

from rsrch import rl


def stack(batch: list[dict | Tensor]):
    if isinstance(batch[0], dict):
        return {k: stack([v[k] for v in batch]) for k in batch[0]}
    else:
        return torch.stack(batch)


def unflatten(batch: dict | Tensor, shape):
    if isinstance(batch, dict):
        return {k: unflatten(batch[k], shape) for k in batch}
    else:
        return batch.reshape(*shape, *batch.shape[1:])


class Data(TypedDict):
    obs: dict | Tensor
    act: dict | Tensor
    reward: Tensor
    term: Tensor


class OffPolicyRLLoader(IterableDataset):
    def __init__(
        self,
        buf: rl.data.Buffer,
        sampler: Sampler,
        sampler_type: Literal["episodes", "slices"],
        batch_size: int,
        slice_len: int,
    ):
        """Create an off-policy RL loader.

        :param buf: RL buffer to use as a data source.
        :param sampler: An index sampler. Depending on `sampler_type`, it should either yield episode IDs, or tuples of episode IDs and slice positions.
        :param sampler_type: See `sampler`.
        :param batch_size: Batch size for the loader.
        :param slice_len: Length (as in, number of observations) for each slice in the batch.
        """

        super().__init__()
        self.buf = buf
        self.sampler = sampler
        self.sampler_type = sampler_type
        self.batch_size = batch_size
        self.slice_len = slice_len

    def empty(self):
        if self.sampler_type == "episodes":
            return all(len(seq) < self.slice_len for seq in self.buf.values())
        else:
            return next(iter(self.sampler), None) is None

    def __iter__(self):
        pos_iter = iter(self.sampler)

        while True:
            batch = []
            while len(batch) < self.batch_size:
                pos = next(pos_iter, None)
                if pos is None:
                    break

                if self.sampler_type == "episodes":
                    ep_id = pos
                    seq = self.buf[ep_id]
                    if len(self) < self.slice_len:
                        continue
                    index = np.random.randint(len(seq) - self.slice_len + 1)
                else:
                    ep_id, index = pos
                    seq = self.buf[ep_id]

                subseq = seq[index : index + self.slice_len]
                batch.append(subseq)

            yield self.collate_fn(batch)

    def collate_fn(self, batch: list[Sequence[dict]]):
        batch = [[*seq] for seq in batch]
        obs, act, reward, term = [], [], [], []
        for t in range(self.slice_len):
            for idx in range(self.batch_size):
                obs.append(batch[idx][t]["obs"])
                term.append(batch[idx][t].get("term", False))
                if t > 0:
                    act.append(batch[idx][t]["act"])
                    reward.append(batch[idx][t]["reward"])

        obs = stack(obs)
        obs = unflatten(obs, (self.slice_len, self.batch_size))
        act = stack(act)
        act = unflatten(act, (self.slice_len - 1, self.batch_size))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        reward = reward.reshape(self.slice_len - 1, self.batch_size)
        term = torch.tensor(np.array(term))
        term = term.reshape(self.slice_len, self.batch_size)

        return Data(obs=obs, act=act, reward=reward, term=term)


class OnPolicyRLLoader(IterableDataset):
    def __init__(
        self,
        do_env_step: Callable[[], tuple[int, tuple[dict, dict]]],
        temp_buf: rl.data.Buffer,
        steps_per_batch: int,
        min_seq_len: int,
    ):
        super().__init__()
        self.do_env_step = do_env_step
        self.temp_buf = temp_buf
        self.steps_per_batch = steps_per_batch
        self.min_seq_len = min_seq_len

    def empty(self):
        return False

    def __iter__(self):
        ep_ids = defaultdict(lambda: None)
        prev_obs = defaultdict(lambda: None)

        while True:
            self.temp_buf.clear()
            ep_ids.clear()

            for env_idx in prev_obs:
                ep_ids[env_idx] = self.temp_buf.reset({"obs": prev_obs[env_idx]})

            for _ in range(self.steps_per_batch):
                env_idx, (step, final) = self.do_env_step()
                ep_ids[env_idx] = self.temp_buf.push(ep_ids[env_idx], step, final)
                prev_obs[env_idx] = step["obs"]
                if final:
                    del ep_ids[env_idx], prev_obs[env_idx]

            batch: list[Data] = []
            for seq in self.temp_buf.values():
                if len(seq) < self.min_seq_len:
                    continue

                obs = stack([step["obs"] for step in seq])
                act = stack([step["act"] for step in seq[1:]])
                reward = torch.tensor(np.array([step["reward"] for step in seq[1:]]))
                term = torch.tensor(np.array([step.get("term", False) for step in seq]))

                batch.append(Data(obs=obs, act=act, reward=reward, term=term))

            yield batch
