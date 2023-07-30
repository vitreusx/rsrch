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
        refresh_every=None,
    ):
        self.size = size
        nlevels = int(np.ceil(np.log2(self.size)))
        self._nleaves = 2**nlevels

        if dtype is None:
            dtype = np.array([zero]).dtype
        self.tree = np.empty((2 * self._nleaves - 1,), dtype=dtype)

        self.reduce_fn = reduce_fn
        self._zero = zero
        self._array_beg = self._nleaves - 1

        self.tree.fill(self._zero)
        if init is None:
            init = self._zero
        self.array.fill(init)
        self.refresh()

        self._refresh_every = refresh_every
        self._refresh_ctr = 0

    def __len__(self):
        return self.size

    @property
    def array(self):
        return self.tree[self._array_beg : self._array_beg + self.size]

    @property
    def total(self):
        return self.tree[0]

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idxes, values):
        idxes, values = self._asarray(idxes, values)
        self.array[idxes] = values
        self._update_tree(idxes, values, 0, self.tree[0], 0, self._nleaves)

        if self._refresh_every is not None:
            self._refresh_ctr += 1
            if self._refresh_ctr >= self._refresh_every:
                self.refresh()
                self._refresh_ctr = 0

    def _asarray(self, idxes, values):
        if isinstance(idxes, slice):
            idxes = np.arange(idxes.start, idxes.stop, idxes.step)
        else:
            idxes = np.asarray(idxes).reshape((-1,))
        values = np.asarray(values).reshape((-1,))
        return idxes, values

    def _update_tree(
        self, idxes: np.ndarray, values: np.ndarray, root: int, root_v, beg, end
    ):
        is_leaf = root >= self._array_beg

        if is_leaf:
            root_v = values[0]
        else:
            left, right = 2 * root + 1, 2 * root + 2
            left_v, right_v = self.tree[left], self.tree[right]
            mid = (beg + end) // 2
            pivot = np.searchsorted(idxes, mid)

            left_idxes, right_idxes = idxes[:pivot], idxes[pivot:]

            if len(left_idxes) > 0:
                left_values = values[:pivot]
                left_v = self._update_tree(
                    left_idxes, left_values, left, left_v, beg, mid
                )

            if len(right_idxes) > 0:
                right_values = values[pivot:]
                right_v = self._update_tree(
                    right_idxes, right_values, right, right_v, mid, end
                )

            root_v = self.reduce_fn(left_v, right_v)

        self.tree[root] = root_v
        return root_v

    def refresh(self):
        for node in reversed(range(self._array_beg)):
            left, right = 2 * node + 1, 2 * node + 2
            left_v, right_v = self.tree[left], self.tree[right]
            node_v = self.reduce_fn(left_v, right_v)
            self.tree[node] = node_v

    def searchsorted(self, values):
        values = np.asarray(values)
        is_scalar = len(values.shape) == 0
        values = values.reshape((-1,))
        idxes = np.empty(values.shape, dtype=np.int32)

        sort_idx = np.argsort(values)
        values = values[sort_idx].copy()

        pivot = np.searchsorted(values, self.tree[0], side="right")
        idxes[pivot:] = len(self)
        values[pivot:] = self.tree[0] - values[pivot:]

        self._search_tree(
            values=values[:pivot],
            offset=self._zero,
            idxes=idxes[:pivot],
            root=0,
            root_v=self.tree[0],
            beg=0,
            end=self._nleaves,
        )
        idxes = idxes[np.argsort(sort_idx)]

        idxes = idxes[0] if is_scalar else idxes
        values = values[0] if is_scalar else values
        return idxes, values

    def _search_tree(self, values, offset, idxes, root, root_v, beg, end):
        is_leaf = root >= self._array_beg

        if is_leaf:
            idxes[:] = beg
            values[:] = offset + root_v - values
        else:
            left, right = 2 * root + 1, 2 * root + 2
            left_v, right_v = self.tree[left], self.tree[right]
            mid = (beg + end) // 2
            pivot = np.searchsorted(values, left_v + offset, side="right")

            left_value, right_value = values[:pivot], values[pivot:]

            if len(left_value) > 0:
                left_offset = offset
                left_idxes = idxes[:pivot]
                self._search_tree(
                    left_value, left_offset, left_idxes, left, left_v, beg, mid
                )

            if len(right_value) > 0:
                right_offset = offset + left_v
                right_idxes = idxes[pivot:]
                self._search_tree(
                    right_value, right_offset, right_idxes, right, right_v, mid, end
                )

    def __repr__(self):
        return f"RangeQueryTree({self.array[:self.size]})"
