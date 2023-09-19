from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class Config:
    enabled: bool
    start: dict
    duration: dict


class Trigger:
    def __init__(self, cfg: dict, step_fn, ref_step=None, ref_t=None):
        self._cfg = cfg
        self._step_fn = step_fn
        self._fired = False
        if ref_step is None:
            ref_step = step_fn()
        self._ref_step = ref_step
        if ref_t is None:
            ref_t = datetime.now()
        self._ref_t = ref_t

    def fire(self):
        self._fired = True

    def __bool__(self):
        if self._fired:
            return False
        if "step" in self._cfg:
            if self._step_fn() - self._ref_step >= self._cfg["step"]:
                return True
        if "time" in self._cfg:
            elapsed = (datetime.now() - self._ref_t).total_seconds()
            if elapsed >= self._cfg["time"]:
                return True
        return False


class Profiler:
    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        step_fn=None,
        trace_path=None,
    ):
        self._cfg = cfg
        self._step_fn = step_fn
        self._trace_path = trace_path

        if self._cfg.enabled:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            self._prof = torch.profiler.profile(
                activities=activities,
                with_stack=True,
            )
            self._start_prof = Trigger(cfg.start, step_fn)
            self._end_prof = None
        else:
            self.update = lambda: None

    def update(self):
        if self._start_prof:
            self._start_prof.fire()
            self._prof = self._prof.__enter__()
            self._end_prof = Trigger(self._cfg.duration, self._step_fn)

        if self._end_prof is not None:
            if self._end_prof:
                self._end_prof.fire()
                self._prof.__exit__(None, None, None)
                if self._trace_path is not None:
                    trace_path = Path(self._trace_path)
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    self._prof.export_chrome_trace(str(trace_path.absolute()))
