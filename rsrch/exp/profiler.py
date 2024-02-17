from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from torch.profiler import ProfilerAction, record_function


class profiler(torch.profiler.profile):
    def __init__(
        self,
        trace_dir: Path,
        device: torch.device,
        *,
        name=None,
        wait=0,
        warmup=16,
        active=4,
        repeat=1,
        enabled=True,
    ):
        if enabled:
            schedule = torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            )
        else:
            schedule = lambda step: ProfilerAction.NONE

        def on_trace_ready(prof: torch.profiler.profile):
            suffix = f"step={prof.step_num}.json"
            fname = f"{name}__{suffix}" if name is not None else suffix
            trace_path = trace_dir / fname
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path.absolute()))

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        super().__init__(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_flops=False,
            with_modules=True,
        )

        self.__enter__()

        self._region_steps = {}
        self._min_step = 0

    def register(self, *regions: str):
        for region in regions:
            self._region_steps[region] = 0

    def region(self, name):
        @contextmanager
        def region_ctx():
            yield
            self._region_steps[name] += 1
            cur_min_step = min(self._region_steps.values())
            while cur_min_step >= self._min_step:
                self._min_step += 1
                super(self.__class__, self).step()

        return region_ctx()
