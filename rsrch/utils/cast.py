import inspect
import types
import typing
from dataclasses import dataclass, fields, is_dataclass
from functools import partial, wraps
from typing import Any, Callable, ParamSpec, Type, TypeVar, get_args, get_origin

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
        if not isinstance(x, dict):
            raise ValueError(f"Cannot convert non-dict to {t}")

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
            raise ValueError(f"{x} is not one of {t_args}")
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


P, R = ParamSpec("R"), TypeVar("R")


def safe_bind(
    func: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[[], R]:
    """Bind a function with args and kwargs in a type-safe manner - `args` and `kwargs` are converted to types as indicated by the parameter annotations."""

    sig = inspect.signature(func)

    type_map = {}
    for name, param in sig.parameters.items():
        arg_type = param.annotation
        if arg_type == inspect._empty:
            arg_type = Any

        if param.kind == param.VAR_POSITIONAL:
            arg_type = list[arg_type]
        elif param.kind == param.VAR_KEYWORD:
            arg_type = dict[str, arg_type]

        type_map[name] = arg_type

    arg = sig.bind(*args, **kwargs)
    arg.apply_defaults()
    value_map = {}
    for name, value in arg.arguments.items():
        value_map[name] = cast(value, type_map[name])

    args, kwargs = [], {}
    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args.append(value_map[name])
        elif param.kind == param.VAR_POSITIONAL:
            args.extend(value_map[name])
        elif param.kind == param.KEYWORD_ONLY:
            kwargs[name] = value_map[name]
        elif param.kind == param.VAR_KEYWORD:
            kwargs.update(value_map[name])

    @wraps(func)
    def wrapped():
        return_value = func(*args, **kwargs)
        if sig.return_annotation != inspect._empty:
            return_value = cast(return_value, sig.return_annotation)
        return return_value

    return wrapped


def argcast(func: Callable[P, R]) -> Callable[P, R]:
    """Create a variant of a function, in which passed arguments are automatically converted to proper types, as indicated with parameter annotations."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return safe_bind(func, *args, **kwargs)()

    return wrapped
