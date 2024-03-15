import platform
from multiprocessing.reduction import ForkingPickler

import ray
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.multiprocessing.reductions import reduce_tensor


def fix_reductions():
    """On Windows and WSL, when reducing CUDA tensors for the first time, ALL
    CUDA tensors are invalidated. Now, it would appear that one way to remedy
    this is to do this once on a dummy CUDA tensor to avoid surprises later on
    in the execution of the program."""

    if (
        platform.system() == "Windows" or "WSL" in platform.platform()
    ) and torch.cuda.is_available():
        x = torch.empty(8, device="cuda")
        reduce_tensor(x)
        del x


def _deserializer(ctor, args, state=None, items=None, kvpairs=None, setstate=None):
    inst = ctor(*args)
    if state is not None:
        if setstate is None:
            setstate = type(state).__setstate__
        setstate(inst, state)
    if items is not None:
        if hasattr(inst, "extend"):
            inst.extend(items)
        else:
            for item in items:
                inst.append(item)
    if kvpairs is not None:
        for k, v in kvpairs:
            inst[k] = v
    return inst


def deserializer(reduce_value):
    return _deserializer(*reduce_value)


def ray_register_extra_reducers():
    """By default, Ray doesn't use custom serializers from torch. This is an
    issue when, for example, we try to send a shared (e.g CUDA) tensor and find
    out it isn't actually the same tensor but a copy. The fix is to fetch the
    extra reducers and register appropriate serializers in Ray."""

    for type, reduce in ForkingPickler._extra_reducers.items():
        ray.util.register_serializer(
            type,
            serializer=reduce,
            deserializer=deserializer,
        )


def fix_all():
    fix_reductions()
    ray_register_extra_reducers()


fix_all()
