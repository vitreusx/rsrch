from functools import wraps
from pathlib import Path

import torch


class Profiler2:
    def __init__(self, dir: Path, device: torch.device):
        self.dir = dir
        self.device = device
        self._profs = []

    def __call__(self, *, name=None, wait=0, warmup=16, active=4, repeat=4):
        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )

        def on_trace_ready(prof: torch.profiler.profile):
            trace_path = self.dir / f"{name}__step={prof.step_num}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path.absolute()))

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        return torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_flops=False,
            with_modules=True,
        )
