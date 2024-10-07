from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import cached_property
from numbers import Number
from threading import RLock
from typing import Callable, Sequence, Tuple, TypeVar, overload

import numpy as np
import torch

_TORCH_FUNCTIONS = {}


class _NOT_FOUND:
    ...


T = TypeVar("T")


class defer_eval(property):
    """This descriptor class acts like @cached_property in tensor-like classes. Cached values are not carried over when creating new objects (e.g. via `stack` or `cat`). Moreover, this descriptor is safe to use when tracing or compiling."""

    def __init__(self, func: Callable[..., T]):
        super().__init__(func)
        self.func = func
        self._prop = cached_property(func)
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self._prop.__set_name__(owner, name)
        super().__set_name__(owner, name)

    def __get__(self, instance, owner=None) -> T:
        if torch.compiler.is_compiling():
            # Behave like `@property`
            return super().__get__(instance, owner)
        else:
            # Behave like `@cached_property`
            return self._prop.__get__(instance, owner)


class Tensorlike:
    """A class for representing complex tensor-like objects.

    Features:

    - Exposes an interface much like torch.Tensor. One can use torch functions such as `torch.cat`, `torch.stack` etc. on tensor-likes, and the output shall be a tensor-like of the same type.

    - User can register tensor fields via `register`. When using operations such as slicing, stacking, concatenating etc., the output is a tensor-like, with
    operations being executed over the tensor fields.

    - When a "child" tensorlike is created, non-Tensor fields are simply copied, *except* for cached properties (see `functools.cached_property`.)
    """

    def __init__(self, shape: torch.Size):
        self._tensors = {}
        self._batched = {}
        self.shape = shape

    def register(self, __name, __value: T, batched=True) -> T:
        """Register a tensor field. The value must be either a torch.Tensor
        or a Tensorlike.
        """

        if hasattr(self, __name):
            raise ValueError(f"Member variable with name '{__name}' already present.")

        setattr(self, __name, __value)

        if isinstance(__value, (torch.Tensor, Tensorlike)):
            self._tensors[__name] = None
            if batched:
                self._batched[__name] = None

        return __value

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
        new = copy.copy(self)
        new.shape = shape
        new._tensors = copy.copy(self._tensors)
        new._batched = copy.copy(self._batched)

        for name in dir(new.__class__):
            val = getattr(new.__class__, name)
            if isinstance(val, defer_eval) and name in new.__dict__:
                # @cached_property / @defer_eval stores true value in __dict__
                delattr(new, name)

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
        for name in self._batched:
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
        for name in self._batched:
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
        for name in self._batched:
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
            for name in repr._batched:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                out_field = getattr(out, name)
                if isinstance(_tensors[0], Tensorlike):
                    _tensors[0]._torch_cat(_tensors, dim, out=out_field)
                else:
                    torch.cat(_tensors, dim, out=out_field)
                setattr(out, name, out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr._batched:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                if isinstance(_tensors[0], Tensorlike):
                    new = _tensors[0]._torch_cat(_tensors, dim)
                else:
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
            for name in repr._batched:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                out_field = getattr(out, name)
                if isinstance(_tensors[0], Tensorlike):
                    _tensors[0]._torch_stack(_tensors, dim, out=out_field)
                else:
                    torch.stack(_tensors, dim, out=out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr._batched:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                if any(x is None for x in _tensors):
                    continue
                if isinstance(_tensors[0], Tensorlike):
                    new = _tensors[0]._torch_stack(_tensors, dim)
                else:
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

        return self._getitem(idx)

    def _getitem(self, idx):
        fields = {}
        shape = None
        for name in self._batched:
            tensor: Tensorlike = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                new = tensor[idx]
                if shape is None:
                    event_dims = len(tensor.shape) - len(self.shape)
                    shape = new.shape[: len(new.shape) - event_dims]
            else:
                new = tensor._getitem(idx)
                if shape is None:
                    shape = new.shape
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
        for name in self._batched:
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
        for name in self._batched:
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
        for name in self._batched:
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
    #     for name in self._batched:
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
        for name in self._batched:
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

    def type_as(self, other: Tensorlike):
        fields = {}
        for name in self._tensors:
            tensor = getattr(self, name)
            fields[name] = tensor.type_as(getattr(other, name))
        return self._new(self.shape, fields)

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self):
        return self.to(device="cuda")

    def pin_memory(self, device=None):
        fields = {}
        for name in self._tensors:
            tensor: torch.Tensor = getattr(self, name)
            fields[name] = tensor.pin_memory(device)
        return self._new(self.shape, fields)

    def __repr__(self):
        fields = {"shape": self.shape}
        for name in self._tensors:
            tensor = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                tensor = f"Tensor(shape={tensor.shape})"

        cls = type(self.__class__.__name__, (), fields)
        cls = dataclass(cls)
        return repr(cls)


__all__ = ["Tensorlike", "defer_eval"]
