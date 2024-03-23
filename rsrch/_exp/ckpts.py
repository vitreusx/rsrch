import copyreg
import os
import pickle
import shutil
import tempfile
import zipfile

import numpy as np


def _from_reduce(ctor, args, state=None, items=None, kvpairs=None, setstate=None):
    x = ctor(*args)
    if state is not None:
        if setstate is None:
            setstate = type(x).__setstate__
        setstate(x, state)
    if items is not None:
        if hasattr(x, "extend"):
            x.extend(items)
        else:
            for item in items:
                x.append(item)
    if kvpairs is not None:
        for k, v in kvpairs:
            x[k] = v
    return x


def reduce(x):
    if hasattr(x, "__getstate__"):
        state = x.__getstate__()
    elif hasattr(x, "__dict__"):
        state = x.__dict__
    else:
        state = x

    if isinstance(state, dict):
        state = {key: reduce(value) for key, value in state.items()}

    return state


def load(x, s):
    if hasattr(x, "__getstate__"):
        x.__setstate__(s)
    elif hasattr(x, "__dict__"):
        for k in x.__dict__:
            setattr(x, k, load(getattr(x, k), s[k]))
    else:
        x = s
    return x


def save_to_zip(obj, f):
    state = reduce(obj)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as temp_f:
        pickle.dump(state, temp_f)
    try:
        with zipfile.ZipFile(f, mode="w") as zf:
            zf.write(temp_f.name, "ckpt.pt")
    finally:
        os.unlink(temp_f.name)


def load_from_zip(obj, f):
    with zipfile.ZipFile(f, mode="r") as zf:
        ckpt_f = zf.open("ckpt.pt")
        state = pickle.load(ckpt_f)
    return load(obj, state)
