import logging
from abc import ABC, abstractmethod
from textwrap import shorten

from colorama import Fore, Style, just_fix_windows_console


class ColorFormatter(logging.Formatter):
    STYLES = {
        logging.FATAL: Style.BRIGHT + Fore.CYAN,
        logging.CRITICAL: Style.BRIGHT + Fore.MAGENTA,
        logging.ERROR: Style.BRIGHT + Fore.RED,
        logging.WARNING: Style.BRIGHT + Fore.YELLOW,
        logging.INFO: Style.RESET_ALL + Fore.WHITE,
        logging.DEBUG: Style.BRIGHT + Fore.BLACK,
    }

    RESET = Style.RESET_ALL

    def __init__(self, *args, **kwargs):
        just_fix_windows_console()
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        record.color_on = self.STYLES[record.levelno]
        record.color_off = self.RESET
        if len(record.name) > 13:
            record.name = record.name[:6] + "~" + record.name[-6:]
        return super().format(record)


def set_log_format(logger: logging.Logger):
    for handler in logger.handlers:
        interactive = hasattr(handler, "stream") and handler.stream.isatty()
        if interactive:
            fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
            formatter = ColorFormatter(fmt)
        else:
            fmt = "%(asctime)s - %(name)-13s - %(levelname)-8s - %(message)s"
            formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)


class LogMixin(ABC):
    @abstractmethod
    def log(self, level: int, msg):
        ...

    def fatal(self, msg):
        return self.log(logging.FATAL, msg)

    def critical(self, msg):
        return self.log(logging.CRITICAL, msg)

    def error(self, msg):
        return self.log(logging.ERROR, msg)

    def warn(self, msg):
        return self.log(logging.WARN, msg)

    def info(self, msg):
        return self.log(logging.INFO, msg)

    def debug(self, msg):
        return self.log(logging.DEBUG, msg)
