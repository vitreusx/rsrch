from tensordict import tensorclass
from torch import Tensor


def _sum_rightmost(value: Tensor, k: int):
    if k > 0:
        value = value.flatten(len(value.shape) - k).sum(-1)
    return value


def distribution(cls):
    """Make a class into a distribution class. Internally, it:
    - wraps the class with @tensorclass;
    - preserves user-defined __init__ (tensorclass __init__ is replaced by __tc_init__);
    - preserves user-defined __repr__.
    If one wants to extend the distribution, one should inherit from <parent>.Unwrapped and use @distribution wrapper again.
    """

    _unwrapped = cls

    prev_init = None
    if cls.__init__ is not object.__init__:
        prev_init = cls.__init__
        if "__init__" in cls.__dict__:
            del cls.__init__

    prev_repr = None
    if cls.__repr__ is not object.__repr__:
        prev_repr = cls.__repr__

    cls = tensorclass(cls)

    if prev_init is not None:
        cls.__tc_init__ = cls.__init__
        cls.__init__ = prev_init

    if prev_repr is not None:
        cls.__repr__ = prev_repr

    cls.Unwrapped = _unwrapped
    return cls
