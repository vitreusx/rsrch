from __future__ import annotations
from .base import *
from .image import *
from .tensor import *
from typing import Any
import torch
from torch import Tensor
import numpy as np
from abc import abstractmethod


_DEFAULT_CAST = {}


def register_default_cast(domain_type: type[Space], codomain_type: type[Space]):
    def decorator(_cls):
        _DEFAULT_CAST[domain_type, codomain_type] = _cls
        return _cls

    return decorator


def default_cast(space: Space, codomain_type: type[Space]):
    for super_ in type(space).__mro__:
        if (super_, codomain_type) in _DEFAULT_CAST:
            return _DEFAULT_CAST[super_, codomain_type](space)


class SpaceTransform:
    def __init__(self, domain: Space, codomain: Space):
        self.domain = domain
        self.codomain = codomain

    @property
    def inv(self) -> SpaceTransform:
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, elem: Any) -> Any:
        ...


@register_default_cast(Box, Image)
class BoxAsImage(SpaceTransform):
    def __init__(self, domain: Box, normalize=None, channels_last=True):
        if normalize is None:
            normalize = np.issubdtype(domain.dtype, np.floating)
        codomain = Image(domain.shape, normalize, channels_last)
        super().__init__(domain, codomain)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return arr


@register_default_cast(Box, TensorBox)
class ToTensorBox(SpaceTransform):
    def __init__(self, domain: Box, device=None, dtype=None):
        codomain = TensorBox(
            low=torch.as_tensor(domain.low),
            high=torch.as_tensor(domain.high),
            shape=torch.Size(domain.shape),
            device=device,
            dtype=dtype,
        )
        super().__init__(domain, codomain)

    def __call__(self, arr: np.ndarray) -> Tensor:
        return torch.as_tensor(
            arr, dtype=self.codomain.dtype, device=self.codomain.device
        )


@register_default_cast(Discrete, TensorDiscrete)
class ToTensorDiscrete(SpaceTransform):
    def __init__(self, domain: Discrete, device=None):
        codomain = TensorDiscrete(n=domain.n, device=device)
        super().__init__(domain, codomain)

    def __call__(self, item: int) -> Tensor:
        return torch.as_tensor(
            item,
            dtype=self.codomain.dtype,
            device=self.codomain.device,
        )


@register_default_cast(Image, Image)
class ImageToImage(SpaceTransform):
    def __init__(self, domain: Image, normalize=False, channels_last=True):
        shape = domain.shape
        if domain._channels_last and not channels_last:
            shape = [shape[2], shape[0], shape[1]]
        elif not domain._channels_last and channels_last:
            shape = [shape[1], shape[2], shape[0]]
        codomain = Image(shape, normalize, channels_last)
        super().__init__(domain, codomain)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.domain._channels_last and not self.codomain._channels_last:
            image = image.transpose(2, 0, 1)
        elif not self.domain._channels_last and self.codomain._channels_last:
            image = image.transpose(1, 2, 0)

        image = np.clip(image, self.codomain.low, self.codomain.high)
        image = image.astype(self.codomain.dtype)
        return image


@register_default_cast(Image, TensorImage)
class ToTensorImage(SpaceTransform):
    def __init__(self, domain: Image, normalize=True, device=None):
        self._inter = ImageToImage(domain, normalize, channels_last=False)
        codomain = TensorImage(
            shape=self._inter.codomain.shape,
            normalized=normalize,
            device=device,
        )
        super().__init__(domain, codomain)

    def __call__(self, image: np.ndarray) -> Tensor:
        image = self._inter(image)
        return torch.as_tensor(
            image,
            dtype=self.codomain.dtype,
            device=self.codomain.device,
        )


@register_default_cast(Space, TensorSpace)
class ToTensor(SpaceTransform):
    def __init__(self, domain: Space, device=None):
        if isinstance(domain, Image):
            t = ToTensorImage(domain, device=device)
        elif isinstance(domain, Box):
            t = ToTensorBox(domain, device=device)
        elif isinstance(domain, Discrete):
            t = ToTensorDiscrete(domain, device=device)
        else:
            raise ValueError(type(domain))

        super().__init__(t.domain, t.codomain)
        self._t = t

    @property
    def inv(self):
        return self._t.inv

    def __call__(self, item: Any) -> Tensor:
        return self._t(item)


class Compose(SpaceTransform):
    def __init__(self, *transforms: SpaceTransform):
        domain = transforms[0].domain
        codomain = transforms[-1].codomain
        super().__init__(domain, codomain)
        self.transforms = transforms

    @property
    def inv(self):
        return Compose(*reversed(self.transforms))

    def __call__(self, elem: Any) -> Any:
        for t in self.transforms:
            elem = t(elem)
        return elem
