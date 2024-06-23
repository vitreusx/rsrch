from __future__ import annotations

import copy
from numbers import Number
from typing import Sequence, Tuple, TypeVar, overload

import numpy as np
import torch

_TORCH_FUNCTIONS = {}


T = TypeVar("T")


class Tensorlike:
    """A class for representing complex tensor-like objects.

    The class exposes an interface much like torch.Tensor. Moreover, one can use torch functions such as torch.cat, torch.stack etc. on tensor-likes, and the output shall be a tensor-like of the same type.

    User can register tensor fields via `register`. When using operations such as slicing, stacking, concatenating etc., the output is a tensor-like, with
    operations being executed over the tensor fields.
    """

    def __init__(self, shape: torch.Size):
        self._fields: set
        super().__setattr__("_fields", set())
        self._tensors = set()
        self._batched = {}
        self._shape = shape

    def __setattr__(self, __name, __value):
        if __name in self._fields:
            if isinstance(__value, (torch.Tensor, Tensorlike)):
                self._tensors.add(__name)
            elif __name in self._tensors:
                self._tensors.pop(__name)
        return super().__setattr__(__name, __value)

    def register(self, __name, __value: T, batched=True) -> T:
        """Register a tensor field.

        A non-tensor may be provided - when assigning to a registered non-tensor, if the setter value is a tensor, the member field becomes a regular tensor field, subject to shape ops etc.
        """

        if hasattr(self, __name):
            raise ValueError(f"Member variable with name '{__name}' already present.")

        super().__setattr__(__name, __value)
        self._fields.add(__name)
        if isinstance(__value, (torch.Tensor, Tensorlike)):
            self._tensors.add(__name)
        self._batched[__name] = batched
        return __value

    @property
    def _batched_tensors(self):
        for __name in self._tensors:
            if self._batched[__name]:
                yield __name

    @property
    def shape(self):
        return self._shape

    @property
    def batch_shape(self):
        return self.shape

    @property
    def device(self):
        any_tensor = next(iter(self._tensors))
        return getattr(self, any_tensor).device

    @property
    def dtype(self):
        any_tensor = next(iter(self._tensors))
        return getattr(self, any_tensor).dtype

    @property
    def _prototype(self):
        return torch.empty([], device=self.device).expand(self.shape)

    def _new(self, shape: torch.Size, fields: dict):
        # copy.copy does not clone Tensors, so, assuming that the other stuff
        # is just metadata, it "should" be fairly efficient.
        new = copy.copy(self)
        new._shape = shape
        new._tensors = copy.copy(self._tensors)
        new._fields = copy.copy(self._fields)
        new._batched = copy.copy(self._batched)
        for name, value in fields.items():
            setattr(new, name, value)
        return new

    @staticmethod
    def torch_function(torch_func):
        def _decorator(impl):
            _TORCH_FUNCTIONS[torch_func] = impl.__name__
            return impl

        return _decorator

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in _TORCH_FUNCTIONS:
            if kwargs is None:
                kwargs = {}
            impl_name = _TORCH_FUNCTIONS[func]
            return getattr(cls, impl_name)(*args, **kwargs)
        else:
            return NotImplemented

    def reshape(self, *shape: int | tuple[int, ...]):
        if isinstance(shape[0], Number):
            return self._reshape(shape)
        else:
            return self._reshape(shape[0])

    def _reshape(self, shape: tuple[int, ...]):
        fields = {}
        new_shape = self.shape
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            elem_shape = tensor.shape[len(self.shape) :]
            new = tensor.reshape(*shape, *elem_shape)
            new_shape = new.shape[: len(new.shape) - len(elem_shape)]
            fields[name] = new
        return self._new(new_shape, fields)

    def flatten(self, start_dim=0, end_dim=-1):
        start_dim = range(len(self.shape))[start_dim]
        end_dim = range(len(self.shape))[end_dim]

        fields = {}
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            new = tensor.flatten(start_dim, end_dim)
            fields[name] = new

        new_shape = [
            *self.shape[:start_dim],
            np.prod(self.shape[start_dim : end_dim + 1]),
            *self.shape[end_dim + 1 :],
        ]
        new_shape = torch.Size(new_shape)

        return self._new(new_shape, fields)

    @torch_function(torch.flatten)
    @classmethod
    def _torch_flatten(cls, tensor, start_dim=0, end_dim=-1):
        return tensor.flatten(start_dim, end_dim)

    @overload
    def expand(self, size: Sequence[int]):
        ...

    def _expand_seq(self, size: Sequence[int]):
        fields = {}
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            event_dims = len(tensor.shape) - len(self.shape)
            new = tensor.expand([*size, *(-1 for _ in range(event_dims))])
            fields[name] = new

        new_shape = self._prototype.expand(size).shape
        return self._new(new_shape, fields)

    @overload
    def expand(self, *sizes: int):
        ...

    def _expand_arg(self, *sizes: int):
        return self._expand_seq([*sizes])

    def expand(self, *args):
        if len(args) == 1 and isinstance(args[0], Sequence):
            return self._expand_seq(args[0])
        else:
            return self._expand_arg(*args)

    @torch_function(torch.cat)
    @classmethod
    def _torch_cat(cls, tensors: Sequence, dim=0, *, out=None):
        repr = tensors[0]
        dim = range(len(repr.shape))[dim]

        new_shape = [*repr.shape]
        new_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        new_shape = torch.Size(new_shape)

        if out is not None:
            for name in repr._batched_tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                out_field = getattr(out, name)
                torch.cat(_tensors, dim, out=out_field)
                setattr(out, name, out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr._batched_tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                new = torch.cat(_tensors, dim)
                fields[name] = new
            return repr._new(new_shape, fields)

    @torch_function(torch.stack)
    @classmethod
    def _torch_stack(cls, tensors: Sequence, dim=0, *, out=None):
        repr = tensors[0]
        if dim < 0:
            dim = len(repr.shape) + dim

        new_shape = [*repr.shape]
        new_shape.insert(dim, len(tensors))
        new_shape = torch.Size(new_shape)

        if out is not None:
            for name in repr._batched_tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                out_field = getattr(out, name)
                torch.stack(_tensors, dim, out=out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr._batched_tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                new = torch.stack(_tensors, dim)
                fields[name] = new
            return repr._new(new_shape, fields)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        num_e = sum(isinstance(x, type(Ellipsis)) for x in idx)
        if num_e > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        if num_e > 0:
            new_idx = []
            for x in idx:
                if isinstance(x, type(Ellipsis)):
                    rem = len(self.shape) - (len(idx) - 1)
                    new_idx.extend(slice(None) for _ in range(rem))
                else:
                    new_idx.append(x)
            idx = tuple(new_idx)

        shape = self._prototype[idx].shape
        return self._getitem(idx, shape)

    def _getitem(self, idx, shape):
        fields = {}
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                new = tensor[idx]
            else:
                new_shape = tuple((*shape, *tensor.shape[len(shape) :]))
                new = tensor._getitem(idx, new_shape)
            fields[name] = new

        return self._new(shape, fields)

    def __setitem__(self, idx, value):
        if not isinstance(idx, tuple):
            idx = (idx,)

        num_e = sum(isinstance(x, type(Ellipsis)) for x in idx)
        if num_e > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        if num_e > 0:
            new_idx = []
            for x in idx:
                if isinstance(x, type(Ellipsis)):
                    rem = len(self.shape) - (len(idx) - 1)
                    new_idx.extend(slice(None) for _ in range(rem))
                else:
                    new_idx.append(x)
            idx = tuple(new_idx)

        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                tensor[idx] = getattr(value, name)
            else:
                tensor._setitem(idx, getattr(value, name))
        return self

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def detach(self):
        fields = {}
        for name in self._tensors:
            tensor: Tensorlike = getattr(self, name)
            fields[name] = tensor.detach()
        return self._new(self.shape, fields)

    @torch_function(torch.detach)
    @classmethod
    def _torch_detach(cls, tensor):
        return tensor.detach()

    def squeeze(self, dim=None):
        if dim is not None:
            if isinstance(dim, int):
                dim = (dim,)
            dim = tuple(range(len(self.shape))[d] for d in dim)

        return self._squeeze(dim=dim)

    def _squeeze(self, dim):
        fields = {}
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                new = tensor.squeeze(dim)
            else:
                new = tensor._squeeze(dim)
            fields[name] = new

        new_shape = self._prototype.squeeze(dim).shape
        return self._new(new_shape, fields)

    @torch_function(torch.squeeze)
    @classmethod
    def _torch_squeeze(cls, tensor, dim=None):
        return tensor.squeeze(dim)

    def unsqueeze(self, dim):
        if dim < 0:
            dim = dim + len(self.shape) + 1

        fields = {}
        for name in self._batched_tensors:
            tensor: Tensorlike = getattr(self, name)
            new = tensor.unsqueeze(dim)
            fields[name] = new

        new_shape = self._prototype.squeeze(dim).shape
        return self._new(new_shape, fields)

    @torch_function(torch.unsqueeze)
    @classmethod
    def _torch_unsqueeze(cls, tensor, dim):
        return tensor.unsqueeze(dim)

    # def where(self, cond, value):
    #     fields = {}
    #     for name in self._batched_tensors:
    #         tensor: Tensorlike = getattr(self, name)
    #         new = tensor.where(cond, value)
    #         fields[name] = new
    #     return self._new(self.shape, fields)

    @overload
    def to(
        self,
        dtype: torch.dtype = None,
        non_blocking=False,
        copy=False,
    ):
        ...

    @overload
    def to(
        self,
        device: torch.Device = None,
        dtype: torch.dtype = None,
        non_blocking=False,
        copy=False,
    ):
        ...

    @overload
    def to(
        self,
        other: Tensorlike,
        non_blocking=False,
        copy=False,
    ):
        ...

    def to(self, *args, **kwargs):
        fields = {}
        for name in self._tensors:
            tensor: Tensorlike = getattr(self, name)
            fields[name] = tensor.to(*args, **kwargs)
        return self._new(self.shape, fields)

    @torch_function(torch.clone)
    @classmethod
    def _torch_clone(cls, tensor):
        return tensor.clone()

    def clone(self):
        fields = {}
        for name in self._tensors:
            tensor: Tensorlike = getattr(self, name)
            fields[name] = tensor.clone()
        return self._new(self.shape, fields)

    def share_memory_(self):
        for name in self._tensors:
            tensor: torch.Tensor = getattr(self, name)
            tensor.share_memory_()
        return self

    @torch_function(torch.gather)
    @classmethod
    def _torch_gather(cls, tensor, dim, index, *, sparse_grad=False):
        return tensor.gather(dim, index, sparse_grad=sparse_grad)

    def gather(self, dim, index, *, sparse_grad=False):
        dim = range(len(self.shape))[dim]
        return self._gather(dim, index, sparse_grad=sparse_grad)

    def _gather(self, dim, index, *, sparse_grad=False):
        fields = {}
        for name in self._batched_tensors:
            tensor: torch.Tensor = getattr(self, name)
            if isinstance(tensor, Tensorlike):
                new = tensor._gather(dim, index, sparse_grad=sparse_grad)
            else:
                event_dim = len(tensor.shape) - len(self.shape)
                new_index = index.reshape(*index.shape, *(1 for _ in range(event_dim)))
                new_index = new_index.expand(
                    *index.shape, *tensor.shape[len(self.shape) :]
                )
                new = tensor.gather(dim, new_index, sparse_grad=sparse_grad)

            fields[name] = new

        new_shape = index.shape
        return self._new(new_shape, fields)

    def type_as(self, other: torch.Tensor):
        fields = {}
        for name in self._tensors:
            tensor: Tensorlike = getattr(self, name)
            fields[name] = tensor.type_as(other)
        return self._new(self.shape, fields)
