import operator as ops

import numpy as np


class RangeQueryTree:
    """A (simplified) range-query tree. We maintain an array :math:`[A_1, \ldots, A_n]`, for which we can (1) query :math:`A_i`, (2) set :math:`A_i := v`, (3) perform equivalent of :func:`np.searchsorted` on :math:`[A_1, A_1 \\oplus A_2, \\ldots, A_1 \\oplus \\ldots \\oplus A_n]` over a specified monoid :math:`M = (\\oplus, 0_{\\oplus})`."""

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
        nlevels = int(np.ceil(np.log2(self.size)))
        self._nleaves = 2**nlevels

        if dtype is None:
            dtype = np.array([zero]).dtype
        self.tree = np.empty((2 * self._nleaves - 1,), dtype=dtype)

        self.reduce_fn = reduce_fn
        self._zero = zero
        self._array_beg = self._nleaves - 1
        self.array = self.tree[self._array_beg : self._array_beg + self.size]

        self.tree.fill(self._zero)
        if init is not None:
            self.array.fill(init)
            for idx in reversed(range(self._array_beg)):
                left_v, right_v = self.tree[2 * idx + 1], self.tree[2 * idx + 2]
                self.tree[idx] = self.reduce_fn(left_v, right_v)

    def __len__(self):
        return self.size

    @property
    def total(self):
        return self.tree[0]

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, np.ndarray):
            value = np.asarray(value)
            value = np.broadcast_to(value, idx.shape)
            for idx_, value_ in zip(idx.ravel(), value.ravel()):
                self[idx_] = value_
            return

        self.array[idx] = value

        node = self._array_beg + idx
        while node > 0:
            node = (node - 1) // 2
            left_v, right_v = self.tree[2 * node + 1], self.tree[2 * node + 2]
            self.tree[node] = self.reduce_fn(left_v, right_v)

    def searchsorted(self, value):
        if isinstance(value, np.ndarray):
            return np.array([self.searchsorted(v) for v in value])

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
        return f"RangeQueryTree({self.array[:self.size]})"
