import logging
from abc import ABC, abstractmethod

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
    def log(self, level: int, msg, *args): ...

    def fatal(self, msg, *args):
        return self.log(logging.FATAL, msg, *args)

    def critical(self, msg, *args):
        return self.log(logging.CRITICAL, msg, *args)

    def error(self, msg, *args):
        return self.log(logging.ERROR, msg, *args)

    def warn(self, msg, *args):
        return self.log(logging.WARN, msg, *args)

    def info(self, msg, *args):
        return self.log(logging.INFO, msg, *args)

    def debug(self, msg, *args):
        return self.log(logging.DEBUG, msg, *args)
