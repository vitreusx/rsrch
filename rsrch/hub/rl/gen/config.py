from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping
from rsrch.utils import cron
from rsrch.utils.cast import safe_partial, typesafe


class Recipe:
    """A class for specifying training recipes.

    The basic idea is that one might want to specify a sequence of actions to perform via YAML config in a following fashion:

    .. code-block:: yaml
        recipe:
        - task1:
            param1: (arg1)
            param2: (arg2)
            ...
        - task2:
            param3: (arg3)
            param4: (arg4)
            ...
        ...

    which is supposed to be executed as:

    .. code-block:: python
        def recipe():
            tasks["task1"](param1=arg1, param2=arg2, ...)
            tasks["task2"](param3=arg3, param4=arg4, ...)
            ...

    with arguments preferably cast to proper types, as specified with type hints.
    """

    def __init__(self, *stages: dict):
        self.stages = stages

    def __call__(
        self,
        fn_store: Mapping[str, Callable],
        cast_types: bool = True,
    ):
        for stage in self.stages:
            assert len(stage) == 1
            fn_name = next(stage.keys())
            params = stage[fn_name]
            assert isinstance(params, dict)
            f = fn_store[fn_name]
            if cast_types:
                f = typesafe(f)
            f(**params)


class _TimeSpan:
    n: float
    of: str | None


TimeSpan = float | _TimeSpan


class Task:
    def __init__(self, every: TimeSpan | None = None, **task: dict):
        assert len(task) == 1
        self.every = every
        self.fn_name = next(task.keys())
        self.fn_params = task[self.fn_name]


class Loop:
    """A class for specifying training loops.

    With a following configuration:

    .. code-block:: yaml
        loop:
          until: (max_time)
          tasks:
            - task1: {param1: (arg1), ...}
              [every: (freq1)]
            - task2: {param2: (arg2), ...}
              [every: (freq2)]
            ...

    we want a following (approximate) behavior:

    .. code-block:: python
        def loop():
            should_run = make_until(until)

            flags = []
            for task in tasks:
                if "every" in tasks:
                    flags.append(make_every(tasks["every"]))
                else:
                    flags.append(None)

            while should_run:
                for task, should_exec in zip(tasks, flags):
                    if should_exec is not None and should_exec:
                        task_fn(**task_params)

    where `make_until` and `make_every` are user-provided functions used to construct the appropriate flags.
    """

    def __init__(self, until: TimeSpan, tasks: list[Task]):
        self.until = until
        self.tasks = tasks

    def __call__(
        self,
        make_until: Callable[[TimeSpan], cron.Flag],
        make_every: Callable[[TimeSpan], cron.Flag],
        fn_store: Mapping[str, Callable],
        cast_types: bool = True,
    ):
        should_run = make_until(self.until)

        flags = []
        for task in self.tasks:
            if task.every is not None:
                flags.append(make_every(self.tasks["every"]))
            else:
                flags.append(None)

        task_fns = []
        for task in self.tasks:
            if cast_types:
                f = safe_partial(fn_store[task.fn_name], **task.fn_params)
            else:
                f = partial(fn_store[task.fn_name], **task.fn_params)
            task_fns.append(task_fn)

        while should_run:
            for task_fn, should_exec in zip(task_fns, flags):
                if should_exec is not None:
                    while should_exec:
                        task_fn()


@dataclass
class Config: ...
