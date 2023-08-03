from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ContextManager, TypeAlias

from torch.profiler import ProfilerAction, ProfilerActivity, profile, schedule

from . import api

SchedFn: TypeAlias = Callable[[int], ProfilerAction]
OnTraceFn: TypeAlias = Callable[[profile], None]


class TorchProfiler(api.Profiler):
    def __init__(self, schedule: SchedFn, on_trace: OnTraceFn, use_cuda=True):
        self.use_cuda = use_cuda

        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)

        self._prof = profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        self._depth = 0
        self._prof.__enter__()

    def __del__(self):
        self._prof.__exit__(None, None, None)

    @staticmethod
    def schedule(wait: int, warmup: int, active: int, repeat: int = 0):
        return schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    @staticmethod
    def on_trace_fn(exp_dir: Path, export_trace=True, export_stack=True):
        def _on_trace(prof: profile):
            if export_trace:
                trace_dest = exp_dir / "traces" / f"trace.step_num={prof.step_num}.json"
                trace_dest.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(trace_dest))

            if export_stack:
                stack_dest = exp_dir / "stacks" / f"stack.step_num={prof.step_num}.json"
                use_cuda = ProfilerActivity.CUDA in prof.activities
                device_type = "cuda" if use_cuda else "cpu"
                stack_dest.parent.mkdir(parents=True, exist_ok=True)
                metric = f"self_{device_type}_time_total"
                prof.export_stacks(str(stack_dest), metric)

        return _on_trace

    @contextmanager
    def profile(self, name: str):
        try:
            self._depth += 1
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._prof.step()
