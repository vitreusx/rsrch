from collections.abc import Mapping


class Buffer(Mapping):
    def __init__(self):
        self.data = {}
        self._next_id = 0

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_id: int):
        return self.data[seq_id]

    def reset(self, obs) -> int:
        seq_id = self._next_id
        self._next_id += 1
        self.data[seq_id] = {"act": [], **{k: [v] for k, v in obs.items()}}
        return seq_id

    def step(self, seq_id: int, act, next_obs):
        seq = self.data[seq_id]
        seq["act"].append(act)
        for k, v in next_obs.items():
            seq[k].append(v)


class Wrapper(Mapping):
    def __init__(self, buf: Buffer):
        self.buf = buf

    def __iter__(self):
        return iter(self.buf)

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, seq_id: int):
        return self.buf[seq_id]

    def reset(self, obs):
        return self.buf.reset(obs)

    def step(self, seq_id: int, act, next_obs):
        self.buf.step(seq_id, act, next_obs)


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
            seq_size = len(self[first_id]["act"]) + 1
            del self[first_id]
            self.size -= seq_size
