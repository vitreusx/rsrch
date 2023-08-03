from abc import ABC, abstractmethod
from functools import wraps
from typing import ContextManager


class Profiler(ABC):
    @abstractmethod
    def profile(self, name: str) -> ContextManager:
        ...

    def wrap(self, func):
        @wraps(func)
        def _wrapped(*args, **kwargs):
            with self.profile(func.__name__):
                return func(*args, **kwargs)

        return _wrapped
