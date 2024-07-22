import operator as ops
from typing import Any

import numpy as np


class rq_tree:
    """A (simplified) range-query tree. We maintain an array :math:`[A_1, \\ldots, A_n]`, for which we can (1) query :math:`A_i`, (2) set :math:`A_i := v`, (3) perform equivalent of :func:`np.searchsorted` on :math:`[A_1, A_1 \\oplus A_2, \\ldots, A_1 \\oplus \\ldots \\oplus A_n]` over a specified monoid :math:`M = (\\oplus, 0_{\\oplus})`."""

    def __init__(
        self,
        size: int,
        reduce_fn=ops.add,
        zero=0.0,
        init=None,
        dtype=None,
    ):
        """Create a range-query tree.

        Args:
            size (int): Size of the underlying array.
            reduce_fn (optional): Associative op for reducing the array. Defaults to ops.add.
            zero (float, optional): Zero element for the reduce_fn. Defaults to 0.0.
            init (_type_, optional): Array initializer. If not specified, defaults to zero of the op.
            dtype (_type_, optional): Numpy dtype for the array. Defaults to None.
        """
        self.size = size
        self.reduce_fn = reduce_fn
        self.zero = zero
        if init is None:
            init = zero
        self.init = init
        if dtype is None:
            dtype = np.array([zero]).dtype
        self.dtype = dtype

        nlevels = int(np.ceil(np.log2(self.size)))
        self._nleaves = 2**nlevels
        self.tree = np.empty((2 * self._nleaves - 1,), dtype=self.dtype)
        self._array_beg = self._nleaves - 1
        self.array = self.tree[self._array_beg : self._array_beg + self.size]

        self.clear()

    def __getstate__(self):
        state = {**self.__dict__}
        del state["reduce_fn"]
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def expand(self, new_size: int):
        if new_size <= self.size:
            return self
        else:
            new_tree = rq_tree(
                new_size, self.reduce_fn, self.zero, self.init, self.dtype
            )
            new_tree.array[: self.size] = self.array[: self.size]
            new_tree._recompute()
            return new_tree

    def __len__(self):
        return self.size

    def clear(self):
        self.array.fill(self.init)
        self._recompute()

    def _recompute(self):
        for idx in reversed(range(self._array_beg)):
            left_v, right_v = self.tree[2 * idx + 1], self.tree[2 * idx + 2]
            self.tree[idx] = self.reduce_fn(left_v, right_v)

    @property
    def total(self):
        """Reduction of entire array (for example, max element for max, or sum for ops.add.)"""
        return self.tree[0]

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idx: int, value: Any):
        self.array[idx] = value
        cur = self._array_beg + idx
        while True:
            cur = (cur - 1) // 2
            left_v, right_v = self.tree[2 * cur + 1], self.tree[2 * cur + 2]
            self.tree[cur] = self.reduce_fn(left_v, right_v)
            if cur == 0:
                break

    def __delitem__(self, idx):
        self[idx] = self.zero

    def searchsorted(self, value):
        if value > self.total:
            return len(self)

        node = 0
        while node < self._array_beg:
            left, right = 2 * node + 1, 2 * node + 2
            left_v = self.tree[left]
            if value <= left_v:
                node, value = left, value
            else:
                node, value = right, value - left_v
        return node - self._array_beg

    def __repr__(self):
        return f"rq_tree({self.array[:self.size]}, total={self.total})"
