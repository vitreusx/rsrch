import abc
import logging
from typing import Any


class Logger(abc.ABC):
    @abc.abstractmethod
    def log(self, level: int, msg: str, *args: Any):
        ...

    def critical(self, msg: str, *args: Any):
        self.log(logging.CRITICAL, msg, *args)

    def error(self, msg: str, *args: Any):
        self.log(logging.ERROR, msg, *args)

    def warning(self, msg: str, *args: Any):
        self.log(logging.WARNING, msg, *args)

    def info(self, msg: str, *args: Any):
        self.log(logging.INFO, msg, *args)

    def debug(self, msg: str, *args: Any):
        self.log(logging.DEBUG, msg, *args)
