from multiprocessing.shared_memory import SharedMemory
import torch.multiprocessing as mp
from collections.abc import MutableMapping
import numpy as np


class shared_ndarray(np.ndarray):
    """An np.ndarray which can be shared between subprocesses."""

    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        shm_name=None,
    ):
        if shm_name is None:
            nbytes = np.dtype(dtype).itemsize * int(np.prod(shape))
            shm = SharedMemory(create=True, size=nbytes)
        else:
            shm = SharedMemory(create=False, name=shm_name)

        buffer = shm.buf
        arr = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
        arr._shm = shm

        return arr
