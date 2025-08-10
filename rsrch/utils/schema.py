import inspect
from inspect import signature
from types import UnionType
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)


def _get_schema(t):
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

    elif t in (list, List):
        if len(t_args) > 0:
            vt = t_args[0]
            return {"type": "array", "items": _get_schema(vt)}
        else:
            return {"type": "array"}

    elif t in (tuple, Tuple):
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
                raise ValueError()
            elif param.kind == param.VAR_KEYWORD:
                if get_origin(param.annotation) in (Any, inspect._empty):
                    additional_properties = True
                else:
                    additional_properties = _get_schema(param.annotation)
            else:
                param_t = param.annotation
                if param_t == inspect._empty:
                    param_t = Any
                properties[param.name] = _get_schema(param_t)
                if param.default == inspect._empty:
                    required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "additionalProperties": additional_properties,
            "required": required,
        }


def get_schema(t: Type, schema_id: str = "https://github.com/vitreusx/rsrch"):
    return {
        "$schema": "https://json-schema.org/draft-07/schema",
        "$id": schema_id,
        **_get_schema(t),
    }
