from collections.abc import Mapping

import numpy as np

from rsrch.types.rq_tree import rq_tree


class Hook:
    def on_create(self, seq_id: int, seq: dict):
        pass

    def on_update(self, seq_id: int, seq: dict):
        pass

    def on_delete(self, seq_id: int, seq: dict):
        pass


class Buffer(Mapping):
    def __init__(self):
        self.data = {}
        self._next_id = 0
        self.hooks: list[Hook] = []

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

    def __delitem__(self, seq_id: int):
        seq = self.data[seq_id]
        for hook in self.hooks:
            hook.on_delete(seq_id, seq)
        del self.data[seq_id]

    def reset(self, obs) -> int:
        seq_id = self._next_id
        self._next_id += 1
        seq = [obs]
        self.data[seq_id] = seq
        for hook in self.hooks:
            hook.on_create(seq_id, seq)
        return seq_id

    def step(self, seq_id: int, act, next_obs):
        seq = self.data[seq_id]
        seq.append({**next_obs, "act": act})
        for hook in self.hooks:
            hook.on_update(seq_id, seq)

    def push(self, seq_id: int | None, step: dict, final: bool):
        if seq_id is None:
            return self.reset(step)
        else:
            step = {**step}
            act = step["act"]
            del step["act"]
            self.step(seq_id, act, step)
            if final:
                seq_id = None
            return seq_id


class Wrapper:
    def __init__(self, buf: Buffer):
        self.buf = buf
        self._unwrapped: Buffer = getattr(buf, "_unwrapped", buf)

    @property
    def hooks(self):
        return self._unwrapped.hooks

    def save(self):
        return self._unwrapped.save()

    def load(self, state):
        self._unwrapped.load(state)

    def __iter__(self):
        return iter(self._unwrapped)

    def __len__(self):
        return len(self._unwrapped)

    def __getitem__(self, seq_id: int):
        return self.buf[seq_id]

    def __delitem__(self, seq_id: int):
        del self.buf[seq_id]

    def reset(self, obs):
        return self.buf.reset(obs)

    def step(self, seq_id, act, next_obs):
        return self.buf.step(seq_id, act, next_obs)

    def push(self, seq_id, step, final):
        return self._unwrapped.push(seq_id, step, final)


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
    def __init__(self, init_size=1024):
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


class Steps(Hook):
    def __init__(self, buf: Buffer):
        super().__init__()
        self.buf = buf
        self.buf.hooks.append(self)
        self._sampler = Sampler()
        self._update()

    def _update(self):
        for seq_id, seq in self.buf.items():
            num_steps = len(seq["act"])
            for offset in range(num_steps):
                self._sampler[(seq_id, offset)] = 1.0

    def on_update(self, seq_id, seq):
        offset = len(seq["act"]) - 1
        self._sampler[(seq_id, offset)] = 1.0

    def on_delete(self, seq_id, seq):
        num_steps = len(seq["act"])
        for offset in range(num_steps):
            del self._sampler[(seq_id, offset)]

    def sample(self):
        seq_id, offset = self._sampler.sample()
        res = self.buf[seq_id][offset + 1]
        next_obs = res["obs"]
        del res["obs"]
        res["obs"] = self.buf[seq_id][offset]["obs"]
        res["next_obs"] = next_obs
        return res


class Slices(Hook):
    def __init__(self, buf: Buffer, slice_len: int):
        super().__init__()
        self.buf = buf
        self.slice_len = slice_len
        self.buf.hooks.append(self)
        self._sampler = Sampler()
        self._update()

    def _update(self):
        for seq_id, seq in self.buf.items():
            num_slices = max(len(seq["act"]) - self.slice_len + 1, 0)
            for offset in range(num_slices):
                self._sampler[(seq_id, offset)] = 1.0

    def on_update(self, seq_id, seq):
        offset = len(seq["act"]) - self.slice_len
        if offset >= 0:
            self._sampler[(seq_id, offset)] = 1.0

    def on_delete(self, seq_id, seq):
        num_slices = max(len(seq["act"]) - self.slice_len + 1, 0)
        for offset in range(num_slices):
            del self._sampler[(seq_id, offset)]

    def sample(self):
        raise NotImplementedError()
