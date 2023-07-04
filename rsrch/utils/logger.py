import logging
from logging import LogRecord
import sys
import os
from colorama import Fore, Style, just_fix_windows_console


class NullLogger:
    def critical(self, msg, *args):
        ...

    def error(self, msg, *args):
        ...

    def warning(self, msg, *args):
        ...

    def info(self, msg, *args):
        ...

    def debug(self, msg, *args):
        ...


class ColorFormatter(logging.Formatter):
    STYLES = {
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


class Logger:
    def __init__(
        self,
        name=None,
        file_path=None,
        stdout_level=logging.WARN,
    ):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.NOTSET)

        stderr = logging.StreamHandler(sys.stderr)
        stderr.setLevel(stdout_level)
        fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
        formatter = ColorFormatter(fmt)
        stderr.setFormatter(formatter)
        self._logger.addHandler(stderr)

        if file_path is not None:
            file = logging.FileHandler(file_path)
            file.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file.setFormatter(formatter)
            self._logger.addHandler(file)

    def critical(self, msg, *args):
        self._logger.critical(msg, *args)

    def error(self, msg, *args):
        self._logger.error(msg, *args)

    def warning(self, msg, *args):
        self._logger.warning(msg, *args)

    def info(self, msg, *args):
        self._logger.info(msg, *args)

    def debug(self, msg, *args):
        self._logger.debug(msg, *args)
