import os
import tempfile
from pathlib import Path
from typing import Protocol

from .data import H5Seq, ListSeq, Sequence


class Store(Protocol):
    def save(self, seq: ListSeq) -> Sequence:
        ...

    def free(self, seq: Sequence):
        ...


class RAMStore(Store):
    def save(self, seq):
        return seq

    def free(self, _):
        pass


class DiskStore(Store):
    def __init__(self, dest_dir: str | os.PathLike):
        self.file_dir = Path(dest_dir)

    def save(self, seq: ListSeq):
        with tempfile.NamedTemporaryFile(
            suffix=".h5", dir=self.file_dir, delete=False
        ) as dest_file:
            dest_path = dest_file.name
        return H5Seq.save(seq, dest_path)

    def free(self, seq: H5Seq):
        del seq._file
        seq._path.unlink()
