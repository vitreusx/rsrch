from abc import ABC, abstractmethod
from logging import *
from typing import Literal

from colorama import Fore, Style, just_fix_windows_console


class ColorFormatter(Formatter):
    STYLES = {
        FATAL: Style.BRIGHT + Fore.CYAN,
        CRITICAL: Style.BRIGHT + Fore.MAGENTA,
        ERROR: Style.BRIGHT + Fore.RED,
        WARNING: Style.BRIGHT + Fore.YELLOW,
        INFO: Style.RESET_ALL + Fore.WHITE,
        DEBUG: Style.BRIGHT + Fore.BLACK,
    }

    RESET = Style.RESET_ALL

    def __init__(self, *args, **kwargs):
        just_fix_windows_console()
        super().__init__(*args, **kwargs)

    def format(self, record: LogRecord) -> str:
        record.color_on = self.STYLES[record.levelno]
        record.color_off = self.RESET
        if len(record.name) > 13:
            record.name = f"{record.name[:6]}~{record.name[-6:]}"
        return super().format(record)


def setup(
    level: int = INFO,
    extra_handlers: list[tuple[Handler, int]] = [],
    no_ansi: bool = False,
):
    logger = getLogger()
    logger.setLevel(DEBUG)

    # Default handler is a StreamHandler to stderr
    err_handler = logger.handlers[0]
    err_handler.setLevel(level)

    for handler, level in extra_handlers:
        handler.setLevel(level)
        logger.addHandler(handler)

    for handler in logger.handlers:
        interactive = hasattr(handler, "stream") and handler.stream.isatty()
        if interactive and not no_ansi:
            fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
            formatter = ColorFormatter(fmt)
        else:
            fmt = "%(asctime)s - %(name)-13s - %(levelname)-8s - %(message)s"
            formatter = Formatter(fmt)
        handler.setFormatter(formatter)


class LogMixin(ABC):
    @abstractmethod
    def log(self, level: int, msg):
        ...

    def fatal(self, msg):
        return self.log(FATAL, msg)

    def critical(self, msg):
        return self.log(CRITICAL, msg)

    def error(self, msg):
        return self.log(ERROR, msg)

    def warn(self, msg):
        return self.log(WARN, msg)

    def info(self, msg):
        return self.log(INFO, msg)

    def debug(self, msg):
        return self.log(DEBUG, msg)
