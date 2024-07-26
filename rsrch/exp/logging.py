from logging import *

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
        return super().format(record)


def setup(
    level: int = INFO,
    extra_handlers: list[tuple[Handler, int]] = [],
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
        if hasattr(handler, "stream") and handler.stream.isatty():
            fmt = "%(name)-13s: %(color_on)s%(levelname)-8s%(color_off)s %(message)s"
            formatter = ColorFormatter(fmt)
        else:
            fmt = "%(asctime)s - %(name)-13s - %(levelname)-8s - %(message)s"
            formatter = Formatter(fmt)
        handler.setFormatter(formatter)
