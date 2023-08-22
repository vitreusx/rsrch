import re
from copy import copy
from pathlib import Path
from types import SimpleNamespace, UnionType
from typing import *

import dacite
import ruamel.yaml as yaml

# A simple ${...} pattern
VAR_PAT = r"\${(?P<expr>[^}]*)}"

# A simple $(...) pattern
EVAL_PAT = r"\$\((?P<expr>[^\)]*)\)"


def quote_preserving_split(sep: str):
    """A pattern for splitting <key>{sep}<value> by {sep}, while respecting
    instances of {sep} in quoted blocks."""
    return f"(?:[\"|'].*?[\"|']|[^({re.escape(sep)})])+"


COMMA_PAT = quote_preserving_split(":")

EQ_PAT = quote_preserving_split("=")


def resolve(data, top_g=None, g_ref=None):
    """Evaluate any $(...) and ${...} expressions in the dictionary."""

    # If starting, initialize top_g (global scope) and g_ref (current scope ref).
    # The global scope is passed to eval for $(...) expressions; current scope
    # ref is used for modifying global scope on the fly.
    if top_g is None:
        top_g = SimpleNamespace()
        g_ref = top_g

    if isinstance(data, dict):
        # Recurse into a dict

        res_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # If field is nested, add an appropriate key to global scope
                # and recurse, passing ref to nested field entry in top_g.
                setattr(g_ref, k, SimpleNamespace())
                res_v = resolve(v, top_g, getattr(g_ref, k))
            else:
                # Otherwise, resolve the value without g_ref and set it
                # afterwards.
                scope = {**top_g.__dict__, **g_ref.__dict__}
                res_v = resolve(v, top_g, scope)
                setattr(g_ref, k, res_v)
            res_data[k] = res_v
        return res_data
    else:
        # Not a dict
        if isinstance(data, str):
            # If it's a string, replace all $(...)'s with eval(..., top_g)
            repl = lambda m: repr(eval(m["expr"], g_ref))
            data = re.sub(VAR_PAT, repl, data)
            data = re.sub(EVAL_PAT, repl, data)
            return data
        else:
            return data


def _cast(x, t):
    """Cast x to t, where t may be a generic (for example Union or Tuple)."""

    # get_origin gets the generic "type", get_args the arguments in square
    # brackets. For example, get_origin(Union[str, int]) == Union and
    # get_args(Union[str, int]) == (str, int)
    orig = get_origin(t)
    if orig is None:
        return x if isinstance(x, t) else t(x)
    elif orig in (Union, Optional, UnionType):
        # Note: get_args(Optional[t]) == (t, None)
        for ti in get_args(t):
            try:
                return _cast(x, ti)
            except:
                pass
        raise ValueError()
    elif orig in (Tuple, List, Set):
        # In this case, since python 3.8, generics are the same as builtins
        # (i.e. Tuple == tuple), so we can use orig for construction
        return orig([_cast(xi, ti) for xi, ti in zip(x, get_args(t))])
    elif orig in (Dict,):
        # Same story with dict
        kt, vt = get_args(t)
        return {k: _cast(xi, vt) for k, xi in x.items()}


def _fix_types(x, t):
    """Align x with type t. Like _cast, but recurses into dataclass fields."""

    if hasattr(x, "__dataclass_fields__"):
        for name, field in x.__dataclass_fields__.items():
            value = getattr(x, name)
            value_t = field.type
            setattr(x, name, _fix_types(value, value_t))
        return x
    else:
        return _cast(x, t)


T = TypeVar("T")


def parse(data: dict, cls: Type[T]) -> T:
    """Parse data into a class. The data may be obtained from e.g. YAML file, and cls must be a dataclass. One can use $(...) expressions for evaluation (like in Bash). Moreover, one can use non-builtin classes in the cls dataclass, which will get converted via the constructor from a base type."""

    # Normalize compound keys
    data = normalize_(copy(data))
    # Resolve all $(...)'s in the data
    data = resolve(data)
    # Use dacite for conversion to the config class. dacite converter
    # throws an error if types mismatch by default, but we fix it afterwards, so
    # we pass check_types=False
    res = dacite.from_dict(cls, data, dacite.Config(check_types=False))
    # Fix the mistyped values, if possible
    res = _fix_types(res, cls)

    return res


def update_(d1: dict, d2: dict):
    """Update d1 with keys and values from d2 in a recursive fashion."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            update_(d1[k], v)
        else:
            d1[k] = v


def merge(dicts: list[dict]):
    """Merge one or more dicts into a single one."""
    merged = {}
    for d in dicts:
        update_(merged, d)
    return merged


def normalize_(d: dict):
    """Normalize a dict, turning any key of form <k1>.<k2>...<kn>: v into k1: {k2: ... kn: v}."""
    for k, v in [*d.items()]:
        if "." in k:
            cur = d
            parts = [*k.split(".")]
            for ki in parts[:-1]:
                if ki not in cur:
                    cur[ki] = {}
                cur = cur[ki]
            cur[parts[-1]] = v
    for k, v in d.items():
        if isinstance(v, dict):
            normalize_(v)
    return d


def parse_var(s: str):
    """Parses a key=value string. Value may be quoted."""
    i = s.find("=")
    k, v = s[:i], s[(i + 1) :]
    k, v = k.strip(), v.strip()
    if v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1]
    return k, v


def read_yml(spec: str, cwd=None):
    """Reads YAML file, either whole or a section. Useful for loading only a part of the file. The format is either <path> or <path>:<section>."""

    if cwd is None:
        cwd = Path.cwd()

    parts = re.findall(COMMA_PAT, spec)
    if len(parts) == 1:
        path = parts[0]
        with open(cwd / path, "r") as f:
            return yaml.safe_load(f)
    elif len(parts) == 2:
        path, sec = parts
        with open(cwd / path, "r") as f:
            data = yaml.safe_load(f)
        for k in sec.split("."):
            data = data[k]
        return data
    else:
        raise ValueError(f'Wrong YAML spec string "{spec}"')
