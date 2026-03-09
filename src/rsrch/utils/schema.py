import inspect
from inspect import signature
from types import UnionType
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)


def _get_schema(t):  # noqa: PLR0911, PLR0912
    t_args = get_args(t)
    t = get_origin(t) or t

    if t in (None, type(None)):
        return {"type": "null"}

    elif t is int:
        return {"type": "integer"}

    elif t is float:
        return {"type": "number"}

    elif t is bool:
        return {"type": "boolean"}

    elif t is str:
        return {"type": "string"}

    elif t == Literal:
        return {"enum": t_args}

    elif t in (list, List):  # noqa: UP006
        if len(t_args) > 0:
            vt = t_args[0]
            return {"type": "array", "items": _get_schema(vt)}
        else:
            return {"type": "array"}

    elif t in (tuple, Tuple):  # noqa: UP006
        if len(t_args) > 0:
            if t_args[-1] == ...:
                return {
                    "type": "array",
                    "minItems": len(t_args) - 1,
                    "prefixItems": [_get_schema(vt) for vt in t_args[:-1]],
                }
            else:
                return {
                    "type": "array",
                    "minItems": len(t_args),
                    "maxItems": len(t_args),
                    "prefixItems": [_get_schema(vt) for vt in t_args],
                }
        else:
            return {"type": "array"}

    elif t in (Union, UnionType, Optional):
        if t == Optional:
            t_args = [*t_args, None]
        return {"anyOf": [_get_schema(vt) for vt in t_args]}

    else:
        sig = signature(t)
        properties = {}
        additional_properties = False
        required = []
        for param in sig.parameters.values():
            if param.kind == param.VAR_POSITIONAL:
                raise ValueError("Constructors with *args list not supported")
            elif param.kind == param.VAR_KEYWORD:
                if get_origin(param.annotation) in (Any, inspect._empty):  # noqa: SLF001
                    additional_properties = True
                else:
                    additional_properties = _get_schema(param.annotation)
            else:
                param_t = param.annotation
                if param_t == inspect._empty:  # noqa: SLF001
                    param_t = Any
                properties[param.name] = _get_schema(param_t)
                if param.default == inspect._empty:  # noqa: SLF001
                    required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": additional_properties,
            "required": required,
        }


def get_schema(t: type, schema_id: str = "https://github.com/vitreusx/rsrch"):
    return {
        "$schema": "https://json-schema.org/draft-07/schema",
        "$id": schema_id,
        **_get_schema(t),
    }
