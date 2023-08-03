from contextlib import contextmanager

from . import api


class NullProfiler(api.Profiler):
    @contextmanager
    def profile(self, name):
        try:
            yield
        finally:
            ...
