from collections import deque
from collections.abc import Mapping
from typing import Iterable, Optional

import numpy as np

from rsrch.utils.vectorize import vectorize

from .data import Seq, Step
from .sampler import Sampler

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
        sampler: Sampler = None,
    ):
        self.chunk_size = chunk_size
        self.step_cap = step_cap
        self.sampler = sampler

        self._step_count = 0
        self._ep_ptr = 0
        self.episodes = deque()
        self._chunk_ptr = 0
        self.chunks = deque()

    def on_reset(self, obs):
        while self._step_count >= self.step_cap:
            self._popleft()
        ep_id = self._ep_ptr
        self._ep_ptr += 1
        ep = Seq([obs], [], [], False)
        self.episodes.append(ep)
        self._step_count += 1
        return ep_id

    def _popleft(self):
        ep_id, offset = self.chunks.popleft()
        ep_idx = len(self.episodes) - (self._ep_ptr - ep_id)
        is_final = offset + self.chunk_size >= len(self.episodes[ep_idx].act)
        if is_final:
            self.episodes.popleft()
        if self.sampler is not None:
            self.sampler.popleft()
        self._step_count -= 1

    def on_step(self, ep_id: int, act, next_obs, reward, term, trunc):
        ep_idx = len(self.episodes) - (self._ep_ptr - ep_id)
        ep = self.episodes[ep_idx]

        while self._step_count >= self.step_cap:
            self._popleft()
        ep.act.append(act)
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
            ...

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

        ep_idx = len(self.episodes) - (self._ep_ptr - ep_id)
        ep = self.episodes[ep_idx]

        size = self.chunk_size
        return Seq(
            obs=ep.obs[off : off + size + 1],
            act=ep.act[off : off + size],
            reward=ep.reward[off : off + size],
            term=ep.term and off + size == len(ep.act),
        )
