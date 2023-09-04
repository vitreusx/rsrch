from abc import ABC, abstractmethod
from functools import wraps
from typing import ContextManager


class Profiler(ABC):
    @abstractmethod
    def profile(self, name: str) -> ContextManager:
        ...

    def annotate(self, _func=None, *, name=None):
        def decorator(func):
            _name = name if name is not None else func.__name__

            @wraps(func)
            def _wrapped(*args, **kwargs):
                with self.profile(_name):
                    return func(*args, **kwargs)

            return _wrapped

        if _func is None:
            return decorator
        else:
            return decorator(_func)
