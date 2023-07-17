from typing import ContextManager, Protocol


class Profiler(Protocol):
    def region(self, name: str) -> ContextManager:
        ...
