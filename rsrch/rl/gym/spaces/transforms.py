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

    def __call__(self, elem: Any) -> Any:
        if isinstance(elem, (Tensor, np.ndarray)):
            elem = elem[None]
        else:
            elem = [elem]
        return self.batch(elem)[0]

    def batch(self, elems: Any) -> Any:
        raise NotImplementedError()


class Identity(SpaceTransform):
    def __init__(self, domain: Space):
        super().__init__(domain, domain)

    @property
    def inv(self):
        return self

    def __call__(self, elem):
        return elem

    def batch(self, elems):
        return elems


class Endomorphism(SpaceTransform):
    def __init__(self, domain: Space, func):
        super().__init__(domain, domain)
        self.func = func

    def __call__(self, elem):
        return self.func(elem)

    def batch(self, elems):
        return self.func(elems)


@register_default_cast(Box, Image)
class BoxAsImage(SpaceTransform):
    def __init__(self, domain: Box, normalize=None, channels_last=True):
        if normalize is None:
            normalize = np.issubdtype(domain.dtype, np.floating)
        codomain = Image(domain.shape, normalize, channels_last)
        super().__init__(domain, codomain)

    def batch(self, arr: np.ndarray) -> np.ndarray:
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

    @property
    def inv(self):
        return FromTensorBox(self.codomain)

    def batch(self, arr: np.ndarray) -> Tensor:
        return torch.as_tensor(
            arr, dtype=self.codomain.dtype, device=self.codomain.device
        )


@register_default_cast(TensorBox, Box)
class FromTensorBox(SpaceTransform):
    def __init__(self, domain: TensorBox):
        codomain = Box(
            low=domain.low.detach().cpu().numpy(),
            high=domain.high.detach().cpu().numpy(),
            shape=domain.shape,
        )
        super().__init__(domain, codomain)

    @property
    def inv(self):
        return ToTensorBox(self.codomain, self.domain.device, self.domain.dtype)

    def batch(self, arr: Tensor) -> np.ndarray:
        return arr.detach().cpu().numpy()


@register_default_cast(Discrete, TensorDiscrete)
class ToTensorDiscrete(SpaceTransform):
    def __init__(self, domain: Discrete, device=None):
        codomain = TensorDiscrete(n=domain.n, device=device)
        super().__init__(domain, codomain)

    @property
    def inv(self):
        return FromTensorDiscrete(self.codomain)

    def batch(self, item: np.ndarray) -> Tensor:
        return torch.as_tensor(
            item,
            dtype=self.codomain.dtype,
            device=self.codomain.device,
        )

    def __call__(self, item: int) -> Tensor:
        return torch.as_tensor(
            item,
            dtype=self.codomain.dtype,
            device=self.codomain.device,
        )


@register_default_cast(TensorDiscrete, Discrete)
class FromTensorDiscrete(SpaceTransform):
    def __init__(self, domain: TensorDiscrete):
        super().__init__(domain, Discrete(domain.n))

    @property
    def inv(self):
        return ToTensorDiscrete(self.codomain, self.domain.device)

    def batch(self, item: Tensor) -> np.ndarray:
        return item.detach().cpu().numpy()

    def __call__(self, item: Tensor) -> int:
        return self.batch(item).item()


@register_default_cast(Image, Image)
class ImageToImage(SpaceTransform):
    def __init__(self, domain: Image, normalize=False, channels_last=True):
        shape = np.array(domain.shape)
        if domain._channels_last and not channels_last:
            shape = [shape[2], shape[0], shape[1]]
        elif not domain._channels_last and channels_last:
            shape = [shape[1], shape[2], shape[0]]
        codomain = Image(shape, normalize, channels_last)
        super().__init__(domain, codomain)

        domain_high = self.domain.high
        if self.domain._channels_last and not self.codomain._channels_last:
            domain_high = domain_high.transpose(2, 0, 1)
        elif not self.domain._channels_last and self.codomain._channels_last:
            domain_high = domain_high.transpose(1, 2, 0)
        self._scale = self.codomain.high / domain_high

    def batch(self, image: np.ndarray) -> np.ndarray:
        if self.domain._channels_last and not self.codomain._channels_last:
            image = image.transpose(0, 3, 1, 2)
        elif not self.domain._channels_last and self.codomain._channels_last:
            image = image.transpose(0, 2, 3, 1)

        image = image * self._scale
        # image = np.clip(image, self.codomain.low, self.codomain.high)
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

    @property
    def inv(self):
        return FromTensorImage(
            self.codomain,
            self.domain._normalize,
            self.domain._channels_last,
        )

    def batch(self, image: np.ndarray) -> Tensor:
        image = self._inter.batch(image)
        return torch.as_tensor(
            image,
            dtype=self.codomain.dtype,
            device=self.codomain.device,
        )


@register_default_cast(TensorImage, Image)
class FromTensorImage(SpaceTransform):
    def __init__(self, domain: TensorImage, normalize=False, channels_last=True):
        _inter = Image(domain.shape, domain._normalized, False)
        self._inter = ImageToImage(_inter, normalize, channels_last)
        super().__init__(domain, self._inter.codomain)

    @property
    def inv(self):
        return ToTensorImage(
            self.codomain,
            self.domain._normalize,
            self.domain.device,
        )

    def batch(self, image: Tensor) -> np.ndarray:
        image = image.detach().cpu().numpy()
        return self._inter.batch(image)


@register_default_cast(Space, TensorSpace)
class ToTensor(SpaceTransform):
    def __init__(self, domain: Space, device=None):
        if isinstance(domain, Image):
            t = ToTensorImage(domain, device=device)
        elif isinstance(domain, Box):
            t = ToTensorBox(domain, device=device)
        elif isinstance(domain, Discrete):
            t = ToTensorDiscrete(domain, device=device)
        elif isinstance(domain, TensorSpace):
            t = Identity(domain)
        else:
            raise ValueError(type(domain))

        super().__init__(t.domain, t.codomain)
        self._t = t

    @property
    def inv(self):
        return self._t.inv

    def __call__(self, item: Any) -> Tensor:
        return self._t(item)

    def batch(self, items: Any) -> Tensor:
        return self._t.batch(items)


class Concat(SpaceTransform):
    def __init__(self, domain: Space, axis=0):
        assert isinstance(domain, Tuple)

        spaces = domain.spaces
        shapes = [np.array(space.shape) for space in spaces]
        new_shape = shapes[0]
        new_shape[axis] = sum(shape[axis] for shape in shapes)

        if isinstance(spaces[0], TensorImage):
            codomain = TensorImage(new_shape, spaces[0]._normalized, spaces[0].device)
        elif isinstance(spaces[0], TensorBox):
            new_low = torch.cat([space.low for space in spaces], axis)
            new_high = torch.cat([space.high for space in spaces])
            codomain = TensorBox(
                new_low, new_high, new_shape, spaces[0].device, spaces[0].dtype
            )
        else:
            raise ValueError(type(spaces[0]))

        super().__init__(domain, codomain)
        self._axis = axis

    def __call__(self, seq) -> Tensor:
        if isinstance(seq[0], Tensor):
            return torch.cat(seq, self._axis)
        else:
            return np.concatenate(seq, self._axis)


class Compose(SpaceTransform):
    def __init__(self, *transforms: SpaceTransform):
        domain = transforms[0].domain
        codomain = transforms[-1].codomain
        super().__init__(domain, codomain)
        self.transforms = []
        for t in transforms:
            if isinstance(t, Compose):
                self.transforms.extend(t.transforms)
            else:
                self.transforms.append(t)

    @property
    def inv(self):
        return Compose(*reversed([t.inv for t in self.transforms]))

    def __call__(self, elem: Any) -> Any:
        for t in self.transforms:
            elem = t(elem)
        return elem

    def batch(self, elems: Any) -> Any:
        for t in self.transforms:
            elems = t.batch(elems)
        return elems


class NormalizeImage(SpaceTransform):
    def __init__(self, domain: Image | TensorImage):
        if isinstance(domain, Image):
            codomain = Image(
                domain.shape, normalized=True, channels_last=domain._channels_last
            )
        else:
            codomain = TensorImage(domain.shape, normalized=True, device=domain.device)
        super().__init__(domain, codomain)
        self._factor = codomain.high[0, 0, 0] / domain.high[0, 0, 0]

    def batch(self, elems):
        return elems * self._factor
