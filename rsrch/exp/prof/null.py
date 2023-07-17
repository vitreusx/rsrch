from contextlib import contextmanager

from . import api


class NullProfiler(api.Profiler):
    @contextmanager
    def region(self, name: str):
        try:
            yield
        finally:
            ...
