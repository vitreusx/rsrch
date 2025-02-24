import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


@contextmanager
def redirect(stream: Literal["stdout", "stderr"], to: str | Path = os.devnull):
    """Redirect stdout or stderr to a given file."""

    fd = getattr(sys, stream).fileno()

    with os.fdopen(os.dup(fd), "w") as old:
        with open(to, "w") as new:
            getattr(sys, stream).close()
            os.dup2(new.fileno(), fd)
            setattr(sys, stream, os.fdopen(fd, "w"))

        try:
            yield
        finally:
            getattr(sys, stream).close()
            os.dup2(old.fileno(), fd)
            setattr(sys, stream, os.fdopen(fd, "w"))
