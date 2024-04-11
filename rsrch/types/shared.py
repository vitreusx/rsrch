import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class shared_ndarray(np.ndarray):
    """A numpy array, which can be shared between processes via pickle."""

    def __new__(cls, shape, dtype=np.float32, shm_name=None):
        if shm_name is None:
            nbytes = np.dtype(dtype).itemsize * int(np.prod(shape))
            shm = SharedMemory(create=True, size=nbytes)
        else:
            shm = SharedMemory(create=False, name=shm_name)

        arr = super().__new__(cls, shape, dtype, shm.buf)
        arr._shm = shm
        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._shm: SharedMemory = getattr(obj, "_shm", None)

    def __repr__(self):
        orig = super().__repr__()
        return orig[:-1] + f", shm_name={self._shm.name})"


def _mp_reducer(arr: shared_ndarray):
    return shared_ndarray, (arr.shape, arr.dtype, arr._shm.name)


ForkingPickler.register(shared_ndarray, _mp_reducer)


def make_shared(arr):
    if isinstance(arr, np.ndarray):
        sh_arr = shared_ndarray(arr.shape, arr.dtype)
        sh_arr[:] = arr
        return sh_arr
    else:
        raise RuntimeError(f"Cannot make {arr} shared.")
