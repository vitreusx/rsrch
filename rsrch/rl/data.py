from collections.abc import Mapping
from typing import Any, MutableMapping

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
        self._unwrapped = getattr(self.buf, "_unwrapped", self.buf)

    @property
    def data(self):
        return self._unwrapped.data

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
    def __init__(self):
        self._key_to_idx, self._next_idx = {}, 0
        self._idx_to_key = []

    def add(self, key):
        index = self._next_idx
        self._next_idx += 1
        self._key_to_idx[key] = index
        self._idx_to_key.append(key)

    def __delitem__(self, key):
        index = self._key_to_idx[key]
        del self._key_to_idx[key]
        self._idx_to_key[index] = None
        if len(self._key_to_idx) / len(self._idx_to_key) < 0.5:
            self._prune()

    def _prune(self):
        keys = [*self._key_to_idx]
        self._key_to_idx, self._next_idx = {}, 0
        for key in keys:
            self.add(key)

    def sample(self):
        while True:
            index = np.random.randint(len(self._idx_to_key))
            if self._idx_to_key[index] is not None:
                return self._idx_to_key[index]

    def __len__(self):
        return len(self._idx_to_key)

    def __iter__(self):
        if len(self._idx_to_key) == 0:
            return
        else:
            while True:
                yield self.sample()


class PSampler:
    def __init__(self, init_size: int = 1024):
        self.prio_tree = rq_tree(init_size)
        self.max_tree = rq_tree(init_size, max)
        self._key_to_idx, self._next_idx = {}, 0
        self._idx_to_key = []

    def __setitem__(self, key, prio_value: float):
        if key not in self._key_to_idx:
            if self._next_idx >= self.prio_tree.size:
                new_size = 2 * self.prio_tree.size
                self.prio_tree = self.prio_tree.expand(new_size)
                self.max_tree = self.max_tree.expand(new_size)
            self._key_to_idx[key] = self._next_idx
            self._idx_to_key.append(key)
            self._next_idx += 1

        key_idx = self._key_to_idx[key]
        self.prio_tree[key_idx] = prio_value
        self.max_tree[key_idx] = prio_value

    def add(self, key):
        if len(self._key_to_idx) == 0:
            prio = 1.0
        else:
            prio = self.max_tree.total
        self[key] = prio

    def __getitem__(self, key):
        key_idx = self._key_to_idx[key]
        return self.prio_tree[key_idx]

    def __delitem__(self, key):
        index = self._key_to_idx[key]
        del self.prio_tree[index]
        del self.max_tree[index]
        del self._key_to_idx[key]
        self._idx_to_key[index] = None

    def sample(self):
        unif = np.random.rand() * self.prio_tree.total
        key_idx = self.prio_tree.searchsorted(unif)
        return self._idx_to_key[key_idx]

    def __len__(self):
        return len(self._idx_to_key)

    def __iter__(self):
        if len(self._idx_to_key) == 0:
            return
        else:
            while True:
                yield self.sample()


class Hook:
    buf: Buffer | None = None

    def on_create(self, seq_id: int):
        pass

    def on_update(self, seq_id: int, time: int):
        pass

    def on_delete(self, seq_id: int):
        pass


class Observable(Wrapper):
    """An 'observable' version of the buffer. One can attach hooks to execute actions when sequences are added, updated or deleted."""

    def __init__(self, buf: Buffer):
        super().__init__(buf)
        self._hooks: list[Hook] = []

    def attach(self, hook: Hook, replay: bool = False):
        """Attach a hook. If `replay` is true, execute actions for each of the sequences in the buffer."""

        if hook.buf is not None:
            raise ValueError("Cannot re-attach a hook.")

        self._hooks.append(hook)
        hook.buf = self
        if replay:
            for seq_id, seq in self.items():
                hook.on_create(seq_id)
                for t in range(1, len(seq)):
                    hook.on_update(seq_id, t)

    def reset(self, obs):
        seq_id = super().reset(obs)
        for hook in self._hooks:
            hook.on_create(seq_id)
        return seq_id

    def step(self, seq_id, act, next_obs):
        super().step(seq_id, act, next_obs)
        seq_len = len(self.data[seq_id])
        for hook in self._hooks:
            hook.on_update(seq_id, seq_len - 1)

    def __delitem__(self, seq_id):
        for hook in self._hooks:
            hook.on_delete(seq_id)
        super().__delitem__(seq_id)


class SliceView(Hook):
    def __init__(self, slice_len: int, sampler: Sampler | PSampler):
        super().__init__()
        self.slice_len = slice_len
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield self.sample()

    def on_update(self, seq_id, time):
        offset = time - (self.slice_len - 1)
        if offset >= 0:
            self.sampler.add((seq_id, offset))

    def on_delete(self, seq_id):
        seq = self.buf[seq_id]
        for time in range(1, len(seq)):
            offset = time - (self.slice_len - 1)
            if offset >= 0:
                del self.sampler[(seq_id, offset)]

    def sample(self):
        seq_id, offset = self.sampler.sample()
        seq = self.buf[seq_id]
        return seq[offset : offset + self.slice_len], (seq_id, offset)
