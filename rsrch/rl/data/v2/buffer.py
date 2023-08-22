from collections import deque
from collections.abc import Mapping
from typing import Iterable, Optional

import numpy as np

from rsrch.utils.vectorize import vectorize

from .data import Seq, Step
from .sampler import Sampler
from .store import RAMStore, Store

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
        step_cap: int,
        frame_stack: int = None,
        sampler: Sampler = None,
        store: Store = None,
    ):
        self.chunk_size = chunk_size
        self.step_cap = step_cap
        self.sampler = sampler
        self.frame_stack = frame_stack

        self._step_count = 0
        self._ep_ptr = 0
        self.episodes = {}
        self._chunk_ptr = 0
        self.chunks = deque()
        self.store = store if store is not None else RAMStore()
        self._store_map = {}

    def on_reset(self, obs):
        while self._step_count >= self.step_cap:
            self._popleft()
        ep_id = self._ep_ptr
        self._ep_ptr += 1

        if self.frame_stack is not None:
            ep = Seq([*obs], [], [], False)
        else:
            ep = Seq([obs], [], [], False)

        self.episodes[ep_id] = ep
        self._step_count += 1
        return ep_id

    def _popleft(self):
        ep_id, offset = self.chunks.popleft()
        is_final = offset + self.chunk_size >= len(self.episodes[ep_id].act)
        if is_final:
            del self.episodes[ep_id]
            if ep_id in self._store_map:
                del self.store[self._store_map[ep_id]]
        if self.sampler is not None:
            self.sampler.popleft()
        self._step_count -= 1

    def on_step(self, ep_id: int, act, next_obs, reward, term, trunc):
        ep = self.episodes[ep_id]

        while self._step_count >= self.step_cap:
            self._popleft()
        ep.act.append(act)
        if self.frame_stack is not None:
            next_obs = next_obs[-1]
        ep.obs.append(next_obs)
        ep.reward.append(reward)
        ep.term = ep.term or term
        self._step_count += 1

        chunk_id = None
        if len(ep.act) >= self.chunk_size:
            chunk_id = self._chunk_ptr
            self._chunk_ptr += 1
            offset = len(ep.act) - self.chunk_size
            self.chunks.append((ep_id, offset))
            if self.sampler is not None:
                self.sampler.append()

        if term or trunc:
            store_id = self.store.add(ep)
            self._store_map[ep_id] = store_id
            self.episodes[ep_id] = self.store[store_id]

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

    def __iter__(self):
        return iter(range(self._chunk_ptr - len(self.chunks), self._chunk_ptr))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, id):
        if isinstance(id, Iterable):
            return [self[i] for i in id]

        idx = len(self.chunks) - (self._chunk_ptr - id)
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
