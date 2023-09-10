from collections import deque
from collections.abc import Mapping
from typing import Iterable, Optional
from copy import deepcopy

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

        step = Step(obs, act, next_obs, reward, term)
        self.steps.append(deepcopy(step))

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
        nsteps: int,
        capacity: int,
        stack_in: int = None,
        stack_out: int = None,
        sampler: Sampler = None,
        persist: Store = None,
    ):
        self.nsteps = nsteps
        self.capacity = capacity
        self.sampler = sampler
        self.stack_in = stack_in
        if stack_out is None and stack_in is not None:
            stack_out = stack_in
        self.stack_out = stack_out

        self.chunks = deque()
        self._chunk_ids = range(0, 0)

        if persist is None:
            persist = {}
        self.episodes = TwoLevelStore(cache={}, persist=persist)
        self._ep_ptr = 0

    def on_reset(self, obs, info):
        ep_id = self._ep_ptr
        self._ep_ptr += 1
        if self.stack_in is not None:
            obs = obs[-1]
        obs = [obs] * (self.stack_out or 1)
        self.episodes[ep_id] = Seq(obs, [], [], False)
        return ep_id

    def _popleft(self):
        ep_id, off = self.chunks.popleft()
        self._chunk_ids = range(self._chunk_ids.start + 1, self._chunk_ids.stop)
        if self.sampler is not None:
            self.sampler.popleft()

        ep = self.episodes[ep_id]
        if off + self.nsteps == len(ep.act):
            del self.episodes[ep_id]

    def on_step(self, ep_id: int, act, next_obs, reward, term, trunc):
        while len(self._chunk_ids) >= self.capacity:
            self._popleft()

        ep = self.episodes[ep_id]
        ep.act.append(act)
        if self.stack_in is not None:
            next_obs = next_obs[-1]
        ep.obs.append(next_obs)
        ep.reward.append(reward)
        ep.term = ep.term or term

        chunk_id = None
        if len(ep.act) >= self.nsteps:
            chunk_id = self._chunk_ids.stop
            self._chunk_ids = range(self._chunk_ids.start, chunk_id + 1)
            offset = len(ep.act) - self.nsteps
            self.chunks.append((ep_id, offset))
            if self.sampler is not None:
                self.sampler.append()

        if term or trunc:
            if len(ep.act) >= self.nsteps:
                self.episodes.persist(ep_id)
            else:
                del self.episodes[ep_id]

        return chunk_id

    def push(self, ep_id: Optional[int], step: Step):
        if ep_id is None:
            ep_id = self.on_reset(step.obs, {})

        chunk_id = self.on_step(
            ep_id, step.act, step.next_obs, step.reward, step.term, step.trunc
        )

        if step.done:
            ep_id = None
        return ep_id, chunk_id

    def __iter__(self):
        return iter(self._chunk_ids)

    def __len__(self):
        return len(self._chunk_ids)

    def __getitem__(self, id):
        if isinstance(id, Iterable):
            return [self[i] for i in id]

        idx = id - self._chunk_ids.start
        ep_id, off = self.chunks[idx]
        ep = self.episodes[ep_id]

        n = self.nsteps
        if self.stack_out is not None:
            obs = []
            for k in range(n + 1):
                obs.append(ep.obs[off + k : off + k + self.stack_out])
        else:
            obs = ep.obs[off : off + n + 1]

        return Seq(
            obs=obs,
            act=ep.act[off : off + n],
            reward=ep.reward[off : off + n],
            term=ep.term and off + n == len(ep.act),
        )


class OnlineBuffer(Mapping[int, Seq]):
    def __init__(self):
        self.episodes = {}
        self._ep_ptr = 0

    def reset(self):
        self.episodes.clear()

    def push(self, ep_id: int, step: Step):
        if ep_id not in self.episodes:
            ep_id = self._ep_ptr
            self._ep_ptr += 1
            self.episodes[ep_id] = Seq([step.obs], [], [], False)

        ep: Seq = self.episodes[ep_id]
        ep.obs.append(step.next_obs)
        ep.act.append(step.act)
        ep.reward.append(step.reward)
        ep.term = ep.term or step.term

        if step.term or step.trunc:
            ep_id = None

        return ep_id

    def __iter__(self):
        return iter(self.episodes)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, id: int) -> Seq:
        return self.episodes[id]
