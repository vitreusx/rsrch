from contextlib import contextmanager

import torch


def to_camel_case(ident: str):
    words = ident.split("_")
    return "".join(word.capitalize() for word in words)


def find_class(module, name):
    return getattr(module, to_camel_case(name))


@contextmanager
def null_ctx():
    yield


def autocast(device=None, dtype=None):
    if dtype is not None:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )
    else:
        return null_ctx()
