import io
import logging
import os
import sys
from logging import LogRecord
from pathlib import Path
from typing import Any, List, Tuple

from colorama import Fore, Style, just_fix_windows_console

from . import api


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


class BaseLogger(api.Logger):
    def __init__(
        self,
        name: str,
        stderr_level=logging.WARN,
        log_file: str | os.PathLike | None = None,
        log_file_level=logging.DEBUG,
    ):
        super().__init__()
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.NOTSET)

        stderr = logging.StreamHandler(sys.stderr)
        stderr.setLevel(stderr_level)
        fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
        formatter = ColorFormatter(fmt)
        stderr.setFormatter(formatter)
        self._logger.addHandler(stderr)

        if log_file is not None:
            file = logging.FileHandler(log_file)
            file.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)-13s - %(levelname)-8ss - %(message)s"
            )
            file.setFormatter(formatter)
            self._logger.addHandler(file)

    def log(self, level: int, msg: str, *args: Any):
        self._logger.log(level, msg, *args)
