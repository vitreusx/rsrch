import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class shared_ndarray(np.ndarray):
    """A numpy array, which can be shared between processes.
    NOTE: Pickles of this class contain shared memory name, NOT the array data.
    To store/load the underlying data, use np.load/np.save."""

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

    def __reduce__(self):
        return shared_ndarray, (self.shape, self.dtype, self._shm.name)

    def __reduce_ex__(self, __protocol):
        return self.__reduce__()

    def __repr__(self):
        orig = super().__repr__()
        return orig[:-1] + f", shm_name={self._shm.name})"
