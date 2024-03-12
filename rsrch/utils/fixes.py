import platform
from multiprocessing.reduction import ForkingPickler

import ray
import torch
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.multiprocessing.reductions import reduce_tensor


def reduce_tensor_fix(tensor: Tensor):
    if tensor.device.type == "cuda" and not getattr(tensor, "_reduced", False):
        data = tensor.detach().cpu()
        reduce_value = reduce_tensor(tensor)
        tensor.copy_(data.to(tensor.device))
        tensor._reduced = True
    else:
        reduce_value = reduce_tensor(tensor)
    return reduce_value


def fix_reductions():
    """On Windows and WSL, when reducing CUDA tensors for the first time,
    the tensor is cleared. The (inefficient, but working) fix is to copy
    the data to CPU (Note: doing .clone() doesn't work), reduce, and then
    reset the tensor to previous value."""

    if platform.system() == "Windows" or "WSL" in platform.platform():
        for type, reduce in ForkingPickler._extra_reducers.items():
            if reduce == reduce_tensor:
                ForkingPickler.register(type, reduce_tensor_fix)


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
