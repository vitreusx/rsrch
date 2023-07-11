from typing import TypeVar

from torch import Tensor

_DETACH_REGISTRY = {}


T = TypeVar("T")


def detach(x: T) -> T:
    if type(x) in _DETACH_REGISTRY:
        _detach_T = _DETACH_REGISTRY[type(x)]
        return _detach_T(x)
    else:
        return x


def register_detach(type_p):
    def _decorator(_func):
        _DETACH_REGISTRY[type_p] = _func
        return _func

    return _decorator


@register_detach(Tensor)
def _detach_tensor(x: Tensor):
    return x.detach()
