import datetime
import os
import socket
from pathlib import Path
from typing import Optional


class ExpDir:
    def __init__(self, root: os.PathLike = "runs", name: Optional[str] = None):
        if name is None:
            now = datetime.datetime.now()
            host = socket.gethostname()
            name = f"{now:%b%d_%H-%M-%S}_{host}"
        self.path = Path(root) / name

    def __truediv__(self, subpath) -> Path:
        return self.path / subpath
