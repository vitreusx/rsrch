import copy
from typing import Sequence, Tuple, overload

import numpy as np
import torch

_TENSORLIKE_TORCH_FUNCTIONS = {}


class Tensorlike:
    def __init__(self, shape: torch.Size):
        super().__setattr__("__fields__", {})
        self._shape = shape
        self.__tensors = {}

    def __getattr__(self, __name):
        if __name in super().__getattribute__("__fields__"):
            return self.__fields__[__name]
        else:
            return super().__getattribute__(__name)

    def __setattr__(self, __name, __value):
        if __name in self.__fields__:
            self.__tensors.pop(__name, None)
            self.register_field(__name, __value)
        else:
            super().__setattr__(__name, __value)

    def register_field(self, __name, __value):
        self.__fields__[__name] = __value
        if isinstance(__value, (torch.Tensor, Tensorlike)):
            self.__tensors[__name] = __value

    @property
    def shape(self):
        return self._shape

    @property
    def batch_shape(self):
        return self.shape

    @property
    def _prototype(self):
        return torch.empty([]).expand(self.shape)

    def _new(self, shape: torch.Size, fields: dict):
        # copy.copy does not clone Tensors, so, assuming that the other stuff
        # is just metadata, it "should" be fairly efficient.
        new = copy.copy(self)
        new._shape = shape
        new.__tensors = copy.copy(self.__tensors)
        new.__fields__ = copy.copy(self.__fields__)
        for name, value in fields.items():
            setattr(new, name, value)
        return new

    @staticmethod
    def torch_function(torch_func):
        def _decorator(impl):
            _TENSORLIKE_TORCH_FUNCTIONS[torch_func] = impl.__name__
            return impl

        return _decorator

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in _TENSORLIKE_TORCH_FUNCTIONS:
            if kwargs is None:
                kwargs = {}
            impl_name = _TENSORLIKE_TORCH_FUNCTIONS[func]
            return getattr(cls, impl_name)(*args, **kwargs)
        else:
            return NotImplemented

    def reshape(self, *shape: int | Tuple[int, ...]):
        fields = {}
        new_shape = torch.Size(*shape)
        for name in self.__tensors:
            tensor: Tensorlike = getattr(self, name)
            elem_shape = tensor.shape[len(self.shape) :]
            tensor = tensor.reshape(*new_shape, *elem_shape)
            fields[name] = tensor
        return self._new(new_shape, fields)

    def flatten(self, start_dim=0, end_dim=-1):
        start_dim = range(len(self.shape))[start_dim]
        end_dim = range(len(self.shape))[end_dim]

        fields = {}
        for name, tensor in self.__tensors.items():
            fields[name] = tensor.flatten(start_dim, end_dim)

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

    def _expand1(self, size: Sequence[int]):
        fields = {}
        for name, tensor in self.__tensors.items():
            event_dims = len(tensor.shape) - len(self.shape)
            tensor = tensor.expand([*size, *(-1 for _ in range(event_dims))])
            fields[name] = tensor

        new_shape = self._prototype.expand(size).shape
        return self._new(new_shape, fields)

    @overload
    def expand(self, *sizes: int):
        ...

    def _expand2(self, *sizes: int):
        return self._expand1([*sizes])

    def expand(self, *args):
        if len(args) == 1 and isinstance(args[0], Sequence):
            return self._expand1(args[0])
        else:
            return self._expand2(*args)

    @torch_function(torch.cat)
    @classmethod
    def _torch_cat(cls, tensors: Sequence, dim=0, *, out=None):
        repr = tensors[0]
        dim = range(len(repr.shape))[dim]

        new_shape = [*repr.shape]
        new_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        new_shape = torch.Size(new_shape)

        if out is not None:
            for name in repr.__tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                out_field = getattr(out, name)
                torch.cat(_tensors, dim, out=out_field)
                setattr(out, name, out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr.__tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                _tensors = torch.cat(_tensors, dim)
                fields[name] = _tensors
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
            for name in repr.__tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                out_field = getattr(out, name)
                torch.stack(_tensors, dim, out=out_field)
            out._shape = new_shape
            return out
        else:
            fields = {}
            for name in repr.__tensors:
                _tensors = [getattr(tensor, name) for tensor in tensors]
                _tensors = torch.stack(_tensors, dim)
                fields[name] = _tensors
            return repr._new(new_shape, fields)

    def at(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        fields = {}
        for name, tensor in self.__tensors.items():
            if isinstance(tensor, torch.Tensor):
                event_dims = len(tensor.shape) - len(self.shape)
                t_idx = (*idx, ..., *(slice(None) for _ in range(event_dims)))
                tensor = tensor[t_idx]
            else:
                tensor = tensor.at(idx)
            fields[name] = tensor

        new_shape = self._prototype[idx].shape

        return self._new(new_shape, fields)

    def __getitem__(self, idx):
        return self.at(idx)

    def __setitem__(self, idx, value):
        for name, tensor in self.__tensors.items():
            if isinstance(tensor, torch.Tensor):
                event_dims = len(tensor.shape) - len(self.shape)
                t_idx = (*idx, ..., *(slice(None) for _ in range(event_dims)))
                tensor[t_idx] = getattr(value, name)
            else:
                tensor[idx] = getattr(value, name)
        return self

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def detach(self):
        fields = {}
        for name, tensor in self.__tensors.items():
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
            dim = (range(len(self.shape))[d] for d in dim)

        fields = {}
        for name, tensor in self.__tensors.items():
            fields[name] = tensor.squeeze(dim)

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
        for name, tensor in self.__tensors.items():
            fields[name] = tensor.unsqueeze(dim)

        new_shape = self._prototype.squeeze(dim).shape
        return self._new(new_shape, fields)

    @torch_function(torch.unsqueeze)
    @classmethod
    def _torch_unsqueeze(cls, tensor, dim):
        return tensor.unsqueeze(dim)
