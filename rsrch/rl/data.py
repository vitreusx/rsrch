from collections.abc import Mapping
from typing import MutableMapping

import numpy as np

from rsrch.types.rq_tree import rq_tree


class Buffer(MutableMapping):
    def __init__(self):
        self.data: dict[int, list] = {}
        self._next_id = 0

    def save(self):
        return (self.data, self._next_id)

    def load(self, state):
        self.data, self._next_id = state

    def __getstate__(self):
        raise RuntimeError(
            "Pickling a buffer is prohibited, in order to prevent accidental "
            "use of torch DataLoader (in which case the different worker processes"
            " would add data to *different* buffers.) Use save()/load() "
            "functions instead."
        )

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_id: int):
        return self.data[seq_id]

    def __setitem__(self, seq_id: int, seq):
        raise RuntimeError("Cannot set buffer sequence directly.")

    def __delitem__(self, seq_id: int):
        del self.data[seq_id]

    def reset(self, obs) -> int:
        seq_id = self._next_id
        self._next_id += 1
        seq = [obs]
        self.data[seq_id] = seq
        return seq_id

    def step(self, seq_id: int, act, step):
        seq = self.data[seq_id]
        next_obs, final = step
        seq.append({**next_obs, "act": act})

    def push(self, seq_id: int | None, step: dict, final: bool):
        if seq_id is None:
            return self.reset(step)
        else:
            step = {**step}
            act = step["act"]
            del step["act"]
            self.step(seq_id, act, (step, final))
            if final:
                seq_id = None
            return seq_id


class Wrapper(MutableMapping):
    def __init__(self, buf: Buffer):
        self.buf = buf

    def save(self):
        return self.buf.save()

    def load(self, state):
        self.buf.load(state)

    def __iter__(self):
        return iter(self.buf)

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, seq_id: int):
        return self.buf[seq_id]

    def __setitem__(self, seq_id: int, seq):
        raise RuntimeError("Cannot set buffer sequence directly.")

    def __delitem__(self, seq_id: int):
        del self.buf[seq_id]

    def reset(self, obs):
        return self.buf.reset(obs)

    def step(self, seq_id, act, next_obs):
        return self.buf.step(seq_id, act, next_obs)

    def push(self, seq_id, step, final):
        if seq_id is None:
            return self.reset(step)
        else:
            step = {**step}
            act = step["act"]
            del step["act"]
            self.step(seq_id, act, (step, final))
            if final:
                seq_id = None
            return seq_id


class SizeLimited(Wrapper):
    def __init__(self, buf: Buffer, cap: int):
        super().__init__(buf)
        self.size, self.cap = 0, cap

    def reset(self, obs):
        seq_id = super().reset(obs)
        self.size += 1
        self._check_size()
        return seq_id

    def step(self, seq_id: int, act, next_obs):
        super().step(seq_id, act, next_obs)
        self.size += 1
        self._check_size()

    def _check_size(self):
        while self.size >= self.cap:
            first_id = next(iter(self))
            seq_size = len(self[first_id])
            del self[first_id]
            self.size -= seq_size


class Sampler:
    def __init__(self, init_size: int = 1024):
        self.prio_tree = rq_tree(init_size)
        self._key_to_idx, self._next_idx = {}, 0
        self._idx_to_key = []

    def __setitem__(self, key, prio_value: float):
        if key not in self._key_to_idx:
            if self._next_idx >= self.prio_tree.size:
                new_size = 2 * self.prio_tree.size
                self.prio_tree = self.prio_tree.expand(new_size)
            self._key_to_idx[key] = self._next_idx
            self._idx_to_key.append(key)
            self._next_idx += 1

        key_idx = self._key_to_idx[key]
        self.prio_tree[key_idx] = prio_value

    def add(self, key):
        self[key] = 1.0

    def __getitem__(self, key):
        key_idx = self._key_to_idx[key]
        return self.prio_tree[key_idx]

    def __delitem__(self, key):
        key_idx = self._key_to_idx[key]
        self.prio_tree[key_idx] = 0.0
        del self._key_to_idx[key]

    def sample(self):
        unif = np.random.rand() * self.prio_tree.total
        key_idx = self.prio_tree.searchsorted(unif)
        return self._idx_to_key[key_idx]

    def __iter__(self):
        while True:
            yield self.sample()


class Hook:
    buf: Buffer | None = None

    def on_create(self, seq_id: int, seq: list):
        pass

    def on_update(self, seq_id: int, seq: list):
        pass

    def on_delete(self, seq_id: int, seq: list):
        pass


class Observable(Wrapper):
    """An 'observable' version of the buffer. One can attach hooks to execute actions when sequences are added, updated or deleted."""

    def __init__(self, buf: Buffer):
        super().__init__(buf)
        self._hooks: list[Hook] = []

    def attach(self, hook: Hook, replay: bool = False):
        """Attach a hook. If `replay`, execute actions for each of the sequences in the buffer."""

        if hook.buf is not None:
            raise ValueError("Cannot re-attach a hook.")

        self._hooks.append(hook)
        hook.buf = self
        if replay:
            for seq_id, seq in self.items():
                hook.on_create(seq_id, seq[:1])
                for t in range(2, len(seq) + 1):
                    hook.on_update(seq_id, seq[:t])

    def reset(self, obs):
        seq_id = super().reset(obs)
        seq = self[seq_id]
        for hook in self._hooks:
            hook.on_create(seq_id, seq)
        return seq_id

    def step(self, seq_id, act, next_obs):
        super().step(seq_id, act, next_obs)
        seq = self[seq_id]
        for hook in self._hooks:
            hook.on_update(seq_id, seq)

    def __delitem__(self, seq_id):
        seq = self[seq_id]
        for hook in self._hooks:
            hook.on_delete(seq_id, seq)
        super().__delitem__(seq_id)


class StepView(Hook):
    def __init__(self):
        super().__init__()
        self._sampler = Sampler()

    def on_update(self, seq_id, seq):
        offset = len(seq) - 1
        self._sampler[(seq_id, offset)] = 1.0

    def on_delete(self, seq_id, seq):
        for offset in range(len(seq) - 1):
            del self._sampler[(seq_id, offset)]

    def sample(self):
        seq_id, offset = self._sampler.sample()
        seq = self.buf[seq_id]
        res = {**seq[offset + 1]}
        obs = res["obs"]
        res["obs"] = seq[offset]["obs"]
        res["next_obs"] = obs
        return res


class SliceView(Hook):
    def __init__(self, slice_len: int):
        super().__init__()
        self.slice_len = slice_len
        self._sampler = Sampler()

    def on_update(self, seq_id, seq):
        offset = len(seq) - self.slice_len
        if offset >= 0:
            self._sampler[(seq_id, offset)] = 1.0

    def on_delete(self, seq_id, seq):
        num_slices = max(len(seq) - self.slice_len + 1, 0)
        for offset in range(num_slices):
            del self._sampler[(seq_id, offset)]

    def sample(self):
        raise NotImplementedError()
