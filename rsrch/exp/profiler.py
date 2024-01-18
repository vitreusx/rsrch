from functools import wraps
from pathlib import Path

import torch


def profiler(
    dir: Path,
    device: torch.device,
    *,
    name=None,
    wait=0,
    warmup=16,
    active=4,
    repeat=4,
):
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat
    )

    def on_trace_ready(prof: torch.profiler.profile):
        suffix = f"step={prof.step_num}.json"
        fname = f"{name}__{suffix}" if name is not None else suffix
        trace_path = dir / fname
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path.absolute()))

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
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
