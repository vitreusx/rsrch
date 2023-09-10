from ..events import *
from typing import Sequence


@dataclass
class VecReset:
    idxes: list[int]
    obs: Any
    info: Sequence[dict]

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, i):
        return Reset(self.idxes[i], self.obs[i], self.info[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class VecStep:
    idxes: list[int]
    act: Any
    next_obs: Any
    reward: Sequence[float]
    term: Sequence[bool]
    trunc: Sequence[bool]
    info: Sequence[dict]

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, i):
        return Step(
            self.idxes[i],
            self.act[i],
            self.next_obs[i],
            self.reward[i],
            self.term[i],
            self.trunc[i],
            self.info[i],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class Async:
    pass


VecEvent = Union[VecReset, VecStep, Async]
