from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

import torch
from torch.profiler import ProfilerActivity, profile, schedule

P, R = ParamSpec("P"), TypeVar("R")


class Profiler:
    def __init__(
        self,
        device: torch.device,
        traces_dir: str | Path,
        schedule: dict,
        options: dict | None = None,
        enabled: bool = True,
    ):
        self.device = device
        self.traces_dir = Path(traces_dir)
        self.schedule = schedule
        if options is None:
            options = dict(with_stack=True, with_modules=True)
        self.options = options
        self.enabled = enabled

        self._activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            self._activities += [ProfilerActivity.CUDA]

    def profiled(
        self,
        _func: Callable[P, R],
        name: str | None = None,
    ) -> Callable[P, R]:
        """Transform a function into a profiled one.

        Transforms a function so as to profile it. To be specific, on the first execution, we repeat the function call until Torch profiler finishes, and save the results to a trace json file. Subsequent calls behave as usual.
        """

        if not self.enabled:
            return _func

        is_finished = False
        if name is None:
            name = _func.__qualname__

        def on_trace_ready(prof: profile):
            dst = self.traces_dir / f"{name}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(dst))

            nonlocal is_finished
            is_finished = True

        @wraps(_func)
        def _wrapped(*args, **kwargs):
            if is_finished:
                return _func(*args, **kwargs)

            with profile(
                activities=self._activities,
                schedule=schedule(**self.schedule),
                on_trace_ready=on_trace_ready,
                **self.options,
            ) as prof:
                while not is_finished:
                    retval = _func(*args, **kwargs)
                    prof.step()
                return retval

        return _wrapped
