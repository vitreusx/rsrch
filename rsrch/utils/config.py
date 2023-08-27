import re
from copy import copy
from pathlib import Path
from types import UnionType
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


class Namespace(dict):
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)


class _Lazy:
    def __init__(self, expr: str, path: list):
        self._expr = expr
        self._path = path

    def eval(self, g: dict):
        cur, local = g, g
        for k in self._path[:-1]:
            cur = cur[k]
            if isinstance(cur, dict):
                local = cur
        g = {**g, **local}
        return eval(self._expr, g)

    def __repr__(self):
        loc = ".".join(str(x) for x in self._path)
        return f"Lazy({loc}: {self._expr})"


def mark_lazy(value, path=[]):
    if isinstance(value, dict):
        new_value = Namespace()
        for k, v in [*value.items()]:
            new_value[k] = mark_lazy(v, [*path, k])
        value = new_value
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = mark_lazy(v, [*path, i])
    elif isinstance(value, str):
        for pat in (VAR_PAT, EVAL_PAT):
            m = re.match(pat, value)
            if m is not None:
                value = _Lazy(m["expr"], path)
                break
    return value


def _resolve_once(data, g):
    lazy = 0
    if isinstance(data, dict):
        for k, v in [*data.items()]:
            new_v, v_lazy = _resolve_once(v, g)
            data[k] = new_v
            lazy += v_lazy
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_v, v_lazy = _resolve_once(v, g)
            data[i] = new_v
            lazy += v_lazy
    elif isinstance(data, _Lazy):
        try:
            data = data.eval(g)
        except:
            pass
        lazy += isinstance(data, _Lazy)
    return data, lazy


def resolve(data):
    data = mark_lazy(data)
    prev_lazy = None
    while prev_lazy != 0:
        data, cur_lazy = _resolve_once(data, data)
        if prev_lazy == cur_lazy:
            break
        prev_lazy = cur_lazy
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
    elif orig in (Tuple, List, Set, tuple, list, set):
        # In this case, since python 3.8, generics are the same as builtins
        # (i.e. Tuple == tuple), so we can use orig for construction
        return orig([_cast(xi, ti) for xi, ti in zip(x, get_args(t))])
    elif orig in (Dict, dict):
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


def parse_data(data: dict, cls: Type[T]) -> T:
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


def parse_var(spec: str):
    """Parses a key=value string. Value may be quoted."""
    parts = re.findall(EQ_PAT, spec)
    if len(parts) > 0:
        k, v = parts[0], parts[1:]
        v = "=".join(v)
        return k, v
    else:
        raise ValueError(spec)


def read_yml(spec: str, cwd=None):
    """Reads YAML file, either whole or a section. Useful for loading only a part of the file. The format is either <path> or <path>:<section>."""

    if cwd is None:
        cwd = Path.cwd()

    parts = re.findall(COMMA_PAT, spec)
    if len(parts) == 1:
        path = parts[0]
        with open(cwd / path, "r") as f:
            return yaml.safe_load(f)
    elif len(parts) > 1:
        path, sec = parts[:-1], parts[-1]
        path = ":".join(path)
        with open(cwd / path, "r") as f:
            data = yaml.safe_load(f)
        for k in sec.split("."):
            data = data[k]
        return data
    else:
        raise ValueError(f'Wrong YAML spec string "{spec}"')


def parse_spec(spec: str, cwd=None):
    try:
        data = read_yml(spec, cwd)
    except:
        k, v = parse_var(spec)
        k = k.split(".")
        data = {}
        cur = data
        for ki in k[:-1]:
            cur[ki] = {}
            cur = cur[ki]
        cur[k[-1]] = v
    return data


def specs_to_data(specs, cwd=None):
    data = {}
    for spec in specs:
        update_(data, parse_spec(spec, cwd))
    return data


def parse(specs: list[str], cls: Type[T] = None, cwd=None) -> dict | T:
    data = specs_to_data(specs, cwd=cwd)
    if cls is not None:
        data = parse_data(data, cls)
    return data
