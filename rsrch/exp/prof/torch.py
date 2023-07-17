from contextlib import contextmanager
from typing import Callable, ContextManager, TypeAlias

from torch.profiler import ProfilerAction, ProfilerActivity, profile, schedule

from . import api

SchedFn: TypeAlias = Callable[[int], ProfilerAction]
OnTraceFn: TypeAlias = Callable[[profile], None]


class TorchProfiler(api.Profiler):
    def __init__(self, on_step: SchedFn, on_trace: OnTraceFn, use_cuda=True):
        self._depth = 0

        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)

        self._prof = profile(
            activities=activities,
            schedule=on_step,
            on_trace_ready=on_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

    @contextmanager
    def region(self, name: str):
        try:
            if self._depth == 0:
                self._depth += 1
                with self._prof as self._prof:
                    yield self._prof
            else:
                self._depth += 1
                yield self._prof
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._prof.step()
