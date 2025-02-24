from enum import Enum
import inspect
import types
import typing
from dataclasses import MISSING, dataclass, fields, is_dataclass
from functools import partial, wraps
from textwrap import indent
from typing import Any, Callable, ParamSpec, Type, TypeVar, get_args, get_origin

T = TypeVar("T")


def cast(x: Any, t: Type[T]) -> T:
    """Cast a value into a given type."""

    orig_t = t
    t_args = get_args(t)
    t = get_origin(t) or t

    if t == Any:
        return x

    elif t in (int, float, str):
        return x if isinstance(x, t) else t(x)

    elif t in (None, type(None)):
        raise ValueError("Value is not None") from None

    elif t in (typing.Union, typing.Optional, types.UnionType):
        for var_t in t_args:
            var_t = get_origin(var_t) or var_t
            if isinstance(var_t, type) and isinstance(x, var_t):
                return x

        errors = []
        for var_t in t_args:
            try:
                return cast(x, var_t)
            except Exception as e:
                errors.append(e)
                pass

        lines = ["Cannot convert value to any of the following types:"]
        for idx, (ti, err) in enumerate(zip(t_args, errors)):
            lines.append(f"(#{idx}) as {ti}:")
            lines.append(indent(str(err), " " * 2))

        raise ValueError("\n".join(lines)) from None

    elif t in (typing.Tuple, tuple):
        x_len, t_len = len(x), len(t_args)
        if t_args[-1] == ...:
            t_args = [*t_args[:-2], *(t_args[-2] for _ in x_len - (t_len - 2))]
            t_len = len(t_args)

        if len(x) != len(t_args):
            raise ValueError(f"Tuple length is incorrect") from None

        values = []
        for idx, (xi, ti) in enumerate(zip(x, t_args)):
            try:
                values.append(cast(xi, ti))
            except Exception as e:
                lines = [
                    f"Casting element #{idx} to {ti} raised error:",
                    indent(str(e), " " * 2),
                ]
                raise ValueError("\n".join(lines)) from None

        return tuple(values)

    elif t in (typing.List, typing.Set, list, set):
        elem_t = t_args[0] if len(t_args) > 0 else Any
        values = []
        for idx, xi in enumerate(x):
            try:
                values.append(cast(xi, elem_t))
            except Exception as e:
                lines = [
                    f"Casting element #{idx} to {elem_t} raised error:",
                    indent(str(e), " " * 2),
                ]
                raise ValueError("\n".join(lines)) from None

        return values if t in (typing.List, list) else {*values}

    elif t in (typing.Dict, dict):
        if len(t_args) > 0:
            kt, vt = t_args
        else:
            kt, vt = Any, Any

        values = {}
        for k, v in x.items():
            try:
                cast_k = cast(k, kt)
            except Exception as e:
                lines = [
                    f"Casting key {k} to {kt} raised error:",
                    indent(str(e), " " * 2),
                ]
                raise ValueError("\n".join(lines)) from None

            try:
                cast_v = cast(v, vt)
            except Exception as e:
                lines = [
                    f"Casting value for '{k}' to {vt} raised error:",
                    indent(str(e), " " * 2),
                ]
                raise ValueError("\n".join(lines)) from None

            values[cast_k] = cast_v

        return values

    elif t == typing.Literal:
        # For Literals, check if the value is one of the allowed values.
        if x not in t_args:
            raise ValueError(f"Value is not one of {t_args}") from None
        return x

    elif issubclass(t, Enum):
        if isinstance(x, str):
            return getattr(t, x)
        else:
            return t(x)

    elif isinstance(t, type) and isinstance(x, t):
        return x

    else:
        sig = inspect.signature(t)

        def get_type_from_ann(ann):
            if ann == inspect._empty:
                return Any
            else:
                return ann

        if isinstance(x, list):
            bound = sig.bind(*x)
        elif isinstance(x, dict):
            bound = sig.bind(**x)
        else:
            bound = sig.bind(x)

        arguments = {}
        for name, value in bound.arguments.items():
            p = sig.parameters[name]
            p_type = get_type_from_ann(p.annotation)
            if p.kind == p.KEYWORD_ONLY:
                value = {k: cast(v, p_type) for k, v in value.items()}
            elif p.kind == p.POSITIONAL_ONLY:
                value = tuple(cast(v, p_type) for v in value)
            else:
                value = cast(value, p_type)
            arguments[name] = value

        bound.arguments = arguments
        return t(*bound.args, **bound.kwargs)


P = ParamSpec("R")
R = TypeVar("R")


def safe_partial(
    func: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Callable[P, R]:
    """Bind a function with args and kwargs in a type-safe manner - `args` and `kwargs` are converted to types as indicated by the parameter annotations.

    A function with the same signature is returned. Positional and keyword arguments provided override the ones given during the binding, and are *not* converted to proper types.
    """

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

    arg = sig.bind_partial(*args, **kwargs)
    args, kwargs = [], {}
    for name, value in arg.arguments.items():
        value = cast(value, type_map[name])
        param = sig.parameters[name]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args.append(value)
        elif param.kind == param.VAR_POSITIONAL:
            args.extend(value)
        elif param.kind == param.KEYWORD_ONLY:
            kwargs[name] = value
        elif param.kind == param.VAR_KEYWORD:
            kwargs.update(value)

    @wraps(func)
    def wrapped(*args2, **kwargs2):
        return_value = func(*args, *args2, **{**kwargs, **kwargs2})
        if sig.return_annotation != inspect._empty:
            return_value = cast(return_value, sig.return_annotation)
        return return_value

    return wrapped


def typesafe(func: Callable[P, R]) -> Callable[P, R]:
    """Create a variant of a function, in which passed arguments are automatically converted to proper types, as indicated with parameter annotations."""

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return safe_partial(func, *args, **kwargs)()

    return wrapped
