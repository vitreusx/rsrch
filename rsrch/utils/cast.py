import inspect
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from typing import Any, Callable, Type, TypeVar, get_args, get_origin

T = TypeVar("T")


def cast(x: Any, t: Type[T]) -> T:
    t_args = get_args(t)
    t = get_origin(t) or t

    if t == Any:
        return x
    elif t in (None, type(None)):
        if x is not None:
            raise ValueError(f"Cannot cast {x} to None.")
        return None
    elif is_dataclass(t):
        args = {}
        field_map = {field.name: field for field in fields(t)}

        for name in x:
            field = field_map[name]
            field_t = field.type
            if isinstance(field_t, str):
                field_t = eval(field_t)
            args[field.name] = cast(x[field.name], field_t)

        return t(**args)
    elif t in (typing.Union, typing.Optional, types.UnionType):
        for ti in t_args:
            ti_ = get_origin(ti) or ti
            if isinstance(ti_, type) and isinstance(x, ti_):
                return x
        for ti in t_args:
            try:
                return cast(x, ti)
            except:
                pass
        raise ValueError(f"None of the variant types {t_args} match value {x}")
    elif t in (typing.Tuple, tuple):
        return tuple([cast(xi, ti) for xi, ti in zip(x, t_args)])
    elif t in (typing.List, typing.Set, list, set):
        elem_t = t_args[0]
        return t([cast(xi, elem_t) for xi in x])
    elif t in (typing.Dict, dict):
        if len(t_args) > 0:
            kt, vt = t_args
        else:
            kt, vt = Any, Any
        return {cast(k, kt): cast(xi, vt) for k, xi in x.items()}
    elif t in (typing.Literal,):
        # For Literals, check if the value is one of the allowed values.
        if x not in t_args:
            raise ValueError(x)
        return x
    elif t == bool and isinstance(x, str):
        x = x.lower()
        if x in ("0", "f", "false", "n", "no"):
            return False
        elif x in ("1", "t", "true", "y", "yes"):
            return True
        else:
            raise ValueError(f"Cannot interpret {x} as bool")
    else:
        return x if isinstance(t, type) and isinstance(x, t) else t(x)


def typed_partial(func: Callable, *args, **kwargs):
    """A variant of partial function from functools, which casts provided positional and keyword arguments to match function signature."""

    sig = inspect.signature(func)

    fields = {"__annotations__": {}}

    for name, param in sig.parameters.items():
        ptype = param.annotation
        if ptype == param.empty:
            ptype = Any
        fields["__annotations__"][name] = ptype

        if param.default != param.empty:
            fields[name] = param.default

    positional = []
    if len(args) > 0:
        for param in sig.parameters.values():
            if param.name not in kwargs:
                positional.append(param.name)

        for name, arg in zip(positional, args):
            fields[name] = arg

    for name, value in kwargs.items():
        fields[name] = value

    dc_type = dataclass(type("_Parameters", (object,), fields))
    params_dc = vars(cast(kwargs, dc_type))

    typed_args = [params_dc[name] for name in positional]
    typed_kwargs = [params_dc[name] for name in kwargs]
    return partial(func, *typed_args, **typed_kwargs)
