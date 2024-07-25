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


def get_logger(
    name: str | None = None,
    handlers: list[tuple[int, logging.Handler] | logging.Handler] = [],
    level: int = logging.INFO,
    no_ansi=False,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in handlers:
        if isinstance(handler, tuple):
            handler, handlerLevel = handler
        else:
            handlerLevel = handler

        handler.setLevel(handlerLevel)
        if no_ansi:
            fmt = "%(asctime)s - %(name)-13s - %(levelname)-8s - %(message)s"
            formatter = logging.Formatter(fmt)
        else:
            fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
            formatter = ColorFormatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
