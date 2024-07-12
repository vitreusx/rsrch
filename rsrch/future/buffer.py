from typing import Any


class SeqBuffer:
    def __init__(self):
        self.data = {}
        self._next_id = 0

    def add_seq(self):
        seq_id = self._next_id
        self.data[seq_id] = {}
        self._next_id += 1
        return seq_id

    def add_step(self, seq_id: int, step: dict[str, Any]):
        seq = self.data[seq_id]
        for k, v in step.items():
            if k not in seq:
                seq[k] = []
            seq[k].append(v)


class SeqView:
    def __init__(self, buf: SeqBuffer):
        ...


class StepView:
    def __init__(self, buf: SeqBuffer):
        ...


class SliceView:
    def __init__(self, buf: SeqBuffer, slice_len: int):
        ...
