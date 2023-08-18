from multiprocessing import shared_memory

import numpy as np
import torch


class shared_ndarray(np.ndarray):
    def __init__(self, shape, dtype, shm_name=None):
        if shm_name is None:
            numel = int(np.prod(shape))
            shm_size = numel * np.dtype(dtype).itemsize
            self._shm = shared_memory.SharedMemory(create=True, size=shm_size)
        else:
            self._shm = shared_memory.SharedMemory(name=shm_name)
        super().__init__(shape, dtype, buffer=self._shm.buf)

    def __reduce__(self):
        args = (self.shape, self.dtype, self._shm.name)
        return (self.__class__, args)


class Storage:
    def __init__(self, shape, dtype, device=None, shared=True):
        if isinstance(dtype, torch.dtype) or isinstance(device, torch.device):
            self._buf = torch.empty(shape, dtype=dtype, device=device)
            if shared:
                self._buf.share_memory_()
        else:
            if shared:
                self._buf = shared_ndarray(shape, dtype)
            else:
                self._buf = np.empty(shape, dtype)

    def __len__(self):
        return len(self._buf)

    def __getitem__(self, idx):
        return self._buf[idx]

    def __setitem__(self, idx, value):
        self._buf[idx] = value


class Cyclic:
    def __init__(self, base):
        self._base = base
        self._size, self._ptr = 0, 0

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return self._base[idx]

    def __setitem__(self, idx, value):
        self._base[idx] = value

    def push(self, x):
        idx = self._ptr
        self._base[self._ptr] = x
        self._size = min(self._size + 1, len(self._base))
        self._ptr = (self._ptr + 1) % len(self._base)
        return idx
