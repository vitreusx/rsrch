from rsrch import spaces

from .. import types


class Buffer(dict[int, dict[str, list]]):
    def __init__(self, spaces: dict[str, spaces.Space]):
        super().__init__()
        self.spaces = spaces
        self.observers = []


class BufferIO:
    def __init__(
        self,
        buf: Buffer,
        capacity: int | None = None,
    ):
        self.buf = buf
        self.capacity = capacity
        self._next_id = 0
        self._observers = []

    def __iter__(self):
        return iter(self.buf)

    def start_seq(self, data: dict) -> int:
        seq_id = self._next_id
        self._next_id += 1
        self.buf[seq_id] = {k: [v] for k, v in data.items()}

    def add_step(self, seq_id: int, data: dict):
        seq = self.buf[seq_id]
        for k, v in data.items():
            if k not in seq:
                seq[k] = []
            seq[k].append(v)

    def stop_seq(self, seq_id: int):
        seq = self.buf[seq_id]
        for k in [*seq]:
            if k in self.buf.spaces:
                space = self.buf.spaces[k]
                seq[k] = space.stack(seq[k])

    def push(
        self,
        ep_id: int | None,
        step: types.Step,
    ):
        if ep_id is None:
            ep_id = self.start_seq({"obs": step.obs})

        self.add_step(
            ep_id,
            {
                "obs": step.next_obs,
                "act": step.act,
                "reward": step.reward,
                "term": step.term,
                "trunc": step.trunc,
            },
        )

        if step.done:
            self.stop_seq(ep_id)
            ep_id = None

        return ep_id


class EpisodeView:
    def __init__(self, buf: Buffer):
        ...
