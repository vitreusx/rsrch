import os
import tempfile
from pathlib import Path
from typing import Protocol

from .data import H5Seq, ListSeq, Sequence, TensorSeq


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


class TensorStore(Store):
    def save(self, seq):
        return TensorSeq.convert(seq)

    def free(self, seq):
        pass


class DiskStore(Store):
    def __init__(self, dest_dir: str | os.PathLike):
        self.file_dir = Path(dest_dir)

    def save(self, seq: ListSeq):
        dest_file = tempfile.NamedTemporaryFile(
            suffix=".h5",
            dir=self.file_dir,
            delete=False,
        )
        return H5Seq.save(seq, Path(dest_file.name))

    def free(self, seq: H5Seq):
        del seq._file
        seq._path.unlink()
