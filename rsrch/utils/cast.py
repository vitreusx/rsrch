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

    elif isinstance(t, type) and isinstance(x, t):
        return x

    else:
        if not isinstance(x, dict):
            raise ValueError(
                "Casting to arbitrary types is supported only for dicts."
            ) from None

        sig = inspect.signature(t)
        params = [*sig.parameters.values()]

        if any(p.kind == p.POSITIONAL_ONLY for p in params):
            raise ValueError(
                f"Casting to {orig_t} is forbidden due to presence of positional-only parameters."
            ) from None

        required = set()
        for name, param in sig.parameters.items():
            if param.default == inspect._empty:
                required.add(name)

        missing = [k for k in required if k not in x]
        if len(missing) > 0:
            if len(missing) == 1:
                missing_list = f"{missing[0]}"
            else:
                missing_list = ", ".join(missing[:-1]) + " and " + missing[-1]
            raise ValueError(
                f"Following parameters are missing: {missing_list}"
            ) from None

        allowed = {*sig.parameters}
        extra = [k for k in x if k not in allowed]
        if len(extra) > 0:
            if len(extra) == 1:
                extra_list = f"{extra[0]}"
            else:
                extra_list = ", ".join(extra[:-1]) + " and " + extra[-1]
            raise ValueError(
                f"Following parameters are superfluous: {extra_list}"
            ) from None

        values = {}
        for name, param in sig.parameters.items():
            if name in x:
                param_type = param.annotation
                if param_type == inspect._empty:
                    param_type = Any
                try:
                    values[name] = cast(x[name], param_type)
                except Exception as e:
                    lines = [
                        f"Casting value for '{name}' to {param_type} raised error:",
                        indent(str(e), " " * 2),
                    ]
                    raise ValueError("\n".join(lines)) from None

        return t(**values)


P = ParamSpec("R")
R = TypeVar("R")


def safe_bind(
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

    arg = sig.bind(*args, **kwargs)
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
        return safe_bind(func, *args, **kwargs)()

    return wrapped
