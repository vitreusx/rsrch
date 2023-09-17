from collections import deque
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from typing import Deque, Iterable, Optional

import numpy as np

from rsrch.rl import gym
from rsrch.rl.gym.vector.utils import concatenate, create_empty_array

from .core import ListSeq, Seq, Step
from .sampler import Sampler
from .store import LayeredStore, RAMStepDeque

__all__ = ["StepBuffer", "ChunkBuffer", "OnlineBuffer"]


class StepBuffer(Mapping[int, Step]):
    def __init__(
        self,
        step_cap: int,
        sampler: Sampler = None,
        store: Deque = None,
    ):
        self.step_cap = step_cap
        self.sampler = sampler

        if store is None:
            store = RAMStepDeque()
        self.steps = store
        self._step_ids = range(0, 0)

    def push(self, obs, act, next_obs, reward, term):
        while len(self._step_ids) >= self.step_cap:
            self.steps.popleft()
            self._step_ids = range(self._step_ids.start + 1, self._step_ids.stop)
            if self.sampler is not None:
                self.sampler.popleft()

        step = Step(obs, act, next_obs, reward, term)
        self.steps.append(deepcopy(step))

        if self.sampler is not None:
            self.sampler.append()
        step_id = self._step_ids.stop
        self._step_ids = range(self._step_ids.start, self._step_ids.stop + 1)

        return step_id

    def __iter__(self):
        return iter(self._step_ids)

    def __len__(self):
        return len(self._step_ids)

    def __getitem__(self, id):
        return self.steps[id - self._step_ids.start]


class ChunkBuffer(Mapping[int, Seq]):
    def __init__(
        self,
        nsteps: int,
        capacity: int,
        obs_space: gym.Space,
        act_space: gym.Space,
        sampler: Sampler = None,
        store: MutableMapping = None,
    ):
        self.nsteps = nsteps
        self.capacity = capacity
        self.sampler = sampler
        self._obs_space = obs_space
        self._act_space = act_space

        self.chunks = deque()
        self._chunk_ids = range(0, 0)

        if store is None:
            store = {}
        self.episodes = LayeredStore(cache={}, persist=store)
        self._ep_ptr = 0

    def on_reset(self, obs, info):
        ep_id = self._ep_ptr
        self._ep_ptr += 1
        self.episodes[ep_id] = ListSeq.initial(obs)
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

        ep: ListSeq = self.episodes[ep_id]
        ep.add(act, next_obs, reward, term, trunc)

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
        ep: Seq = self.episodes[ep_id]

        n = self.nsteps
        obs = create_empty_array(self._obs_space, n + 1)
        concatenate(self._obs_space, ep.obs[off : off + n + 1], obs)
        act = create_empty_array(self._act_space, n)
        concatenate(self._act_space, ep.act[off : off + n], act)
        reward = np.asarray(ep.reward[off : off + n])
        term = ep.term and off + n == len(ep.act)

        return Seq(obs, act, reward, term)


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
