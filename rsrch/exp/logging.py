import logging
from logging import LogRecord

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

    def format(self, record: LogRecord) -> str:
        record.color_on = self.STYLES[record.levelno]
        record.color_off = self.RESET
        return super().format(record)


class _LoggerMixin:
    def log(self, level: int, msg: str, *args):
        raise NotImplementedError()

    def debug(self, msg: str, *args):
        self.log(logging.DEBUG, msg, *args)

    def info(self, msg: str, *args):
        self.log(logging.INFO, msg, *args)

    def warning(self, msg: str, *args):
        self.log(logging.WARNING, msg, *args)

    def warn(self, msg: str, *args):
        self.warning(msg, *args)

    def error(self, msg: str, *args):
        self.log(logging.ERROR, msg, *args)

    def critical(self, msg: str, *args):
        self.log(logging.CRITICAL, msg, *args)

    def fatal(self, msg: str, *args):
        self.log(logging.FATAL, msg, *args)


class Logger(_LoggerMixin):
    def __init__(self, name: str, stream=None, level=logging.INFO, no_ansi=True):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        if no_ansi:
            fmt = "%(asctime)s - %(name)-13s - %(levelname)-8s - %(message)s"
            formatter = logging.Formatter(fmt)
        else:
            fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
            formatter = ColorFormatter(fmt)
        handler.setFormatter(formatter)

        self._logger.addHandler(handler)

    def log(self, level: int, msg: str, *args):
        self._logger.log(level, msg, *args)
