import datetime
import os
import socket
from pathlib import Path
from typing import Optional


class ExpDir:
    def __init__(self, root: os.PathLike = "runs", name: Optional[str] = None):
        if name is None:
            now = datetime.datetime.now()
            name = f"{now:%Y-%m-%d_%H-%M-%S}"
        self.path = Path(root) / name

    def __truediv__(self, subpath) -> Path:
        return self.path / subpath
