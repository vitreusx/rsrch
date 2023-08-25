from collections import deque
from collections.abc import Mapping
from typing import Iterable, Optional


from .data import Seq, Step
from .sampler import Sampler
from .store import Store, TwoLevelStore

__all__ = ["StepBuffer", "ChunkBuffer"]


class StepBuffer(Mapping[int, Step]):
    def __init__(self, step_cap: int, sampler: Sampler = None):
        self.step_cap = step_cap
        self.sampler = sampler

        self.steps = deque()
        self._step_ptr = 0

    def push(self, obs, act, next_obs, reward, term):
        while len(self.steps) >= self.step_cap:
            self.steps.popleft()
            if self.sampler is not None:
                self.sampler.popleft()

        self.steps.append(Step(obs, act, next_obs, reward, term))
        if self.sampler is not None:
            self.sampler.append()
        step_id = self._step_ptr
        self._step_ptr += 1

        return step_id

    def __iter__(self):
        return iter(range(self._step_ptr - len(self.steps), self._step_ptr))

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, id):
        if isinstance(id, Iterable):
            return [self[i] for i in id]

        idx = len(self.steps) - (self._step_ptr - id)
        return self.steps[idx]


class ChunkBuffer(Mapping[int, Seq]):
    def __init__(
        self,
        chunk_size: int,
        capacity: int,
        frame_stack: int = None,
        sampler: Sampler = None,
        persist: Store = None,
    ):
        self.chunk_size = chunk_size
        self.capacity = capacity
        self.sampler = sampler
        self.frame_stack = frame_stack

        self.chunks = deque()
        self._chunk_ids = range(0, 0)

        if persist is None:
            persist = {}
        self.episodes = TwoLevelStore(cache={}, persist=persist)
        self._ep_ptr = 0

    def on_reset(self, obs):
        ep_id = self._ep_ptr
        self._ep_ptr += 1

        obs = [obs] * (self.frame_stack or 1)
        self.episodes[ep_id] = Seq(obs, [], [], False)
        return ep_id

    def _popleft(self):
        ep_id, off = self.chunks.popleft()
        self._chunk_ids = range(self._chunk_ids.start + 1, self._chunk_ids.stop)
        if self.sampler is not None:
            self.sampler.popleft()

        ep = self.episodes[ep_id]
        if off + self.chunk_size == len(ep.act):
            del self.episodes[ep_id]

    def on_step(self, ep_id: int, act, next_obs, reward, term, trunc):
        while len(self._chunk_ids) >= self.capacity:
            self._popleft()

        ep = self.episodes[ep_id]
        ep.act.append(act)
        ep.obs.append(next_obs)
        ep.reward.append(reward)
        ep.term = ep.term or term

        chunk_id = None
        if len(ep.act) >= self.chunk_size:
            chunk_id = self._chunk_ids.stop
            self._chunk_ids = range(self._chunk_ids.start, chunk_id + 1)
            offset = len(ep.act) - self.chunk_size
            self.chunks.append((ep_id, offset))
            if self.sampler is not None:
                self.sampler.append()

        if term or trunc:
            if len(ep.act) >= self.chunk_size:
                self.episodes.persist(ep_id)
            else:
                del self.episodes[ep_id]

        return chunk_id

    def push(self, ep_id: Optional[int], step: Step):
        if ep_id is None:
            ep_id = self.on_reset(step.obs)

        chunk_id = self.on_step(
            ep_id, step.act, step.next_obs, step.reward, step.term, step.trunc
        )

        if step.done:
            ep_id = None
        return ep_id, chunk_id

    def keys(self):
        return self._chunk_ids

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, id):
        if isinstance(id, Iterable):
            return [self[i] for i in id]

        idx = id - self._chunk_ids.start
        ep_id, off = self.chunks[idx]
        ep = self.episodes[ep_id]

        size = self.chunk_size
        if self.frame_stack is None:
            obs = ep.obs[off : off + size + 1]
        else:
            obs = []
            for k in range(size + 1):
                obs.append(ep.obs[off + k : off + k + self.frame_stack])

        return Seq(
            obs=obs,
            act=ep.act[off : off + size],
            reward=ep.reward[off : off + size],
            term=ep.term and off + size == len(ep.act),
        )
