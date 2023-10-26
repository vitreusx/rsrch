from __future__ import annotations

import argparse
import io
import math
import re
from copy import copy
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import *

import dacite
import ruamel.yaml as yaml
from mako.template import Template

from rsrch.types import Namespace

# Either $(...) or ${...}, covering entire text
EVAL_PATS = [r"^\$\((?P<expr>.*)\)$", r"^\${(?P<expr>.*)}$"]


def quote_preserving_split(sep: str):
    """A pattern for splitting <key>{sep}<value> by {sep}, while respecting
    instances of {sep} in quoted blocks."""
    return f"(?:[\"|'].*?[\"|']|[^({re.escape(sep)})])+"


COLON_SEP = quote_preserving_split(":")

EQ_PAT = quote_preserving_split("=")


class Expr:
    """An expression object in the config."""

    def __init__(self, value, path: list):
        self._value = value
        self._path = path
        self._eval_fn = self._make_eval_fn()

    def _make_eval_fn(self):
        # If the field value is ${...} or $(...), we pass the inner text to
        # Python's eval.
        for pat in EVAL_PATS:
            m = re.match(pat, self._value)
            if m is not None:
                return lambda g: eval(m["expr"], g)
        # Otherwise, we use Mako.
        return lambda g: Template(self._value).render(**g)

    def eval(self, g: dict):
        """Evaluate the expression, under a given context."""
        cur, local = g, g

        # Locate field's local scope, i.e. neighbors, and add it to context.
        for k in self._path[:-1]:
            cur = cur[k]
            if isinstance(cur, dict):
                local = cur
        g = {**g, **local}

        if "buffer" in g:
            # Due to a limitation in Mako, we cannot pass "buffer" variable
            # directly, if it is defined.
            assert "_buffer" not in g
            g["_buffer"] = g["buffer"]
            del g["buffer"]

        return self._eval_fn(g)

    def __repr__(self):
        loc = ".".join(str(x) for x in self._path)
        return f"Lazy({loc}: {self._value})"


def to_expr(value, path=[]):
    """Convert string fields in the dict to Expr objects."""

    if isinstance(value, dict):
        new_value = Namespace()
        for k, v in [*value.items()]:
            new_value[k] = to_expr(v, [*path, k])
        value = new_value
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = to_expr(v, [*path, i])
    elif isinstance(value, str):
        value = Expr(value, path)
    return value


def _resolve_once(data, g):
    """Try to convert `Expr` objects to regular values, using current context."""

    num_exprs = 0
    if isinstance(data, dict):
        for k, v in [*data.items()]:
            new_v, v_lazy = _resolve_once(v, g)
            data[k] = new_v
            num_exprs += v_lazy
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_v, v_lazy = _resolve_once(v, g)
            data[i] = new_v
            num_exprs += v_lazy
    elif isinstance(data, Expr):
        try:
            data = data.eval(g)
        except:
            pass
        num_exprs += isinstance(data, Expr)
    return data, num_exprs


def resolve_exprs(data):
    """Convert $(...)'s and ${...}'s to regular values."""
    data = to_expr(data)
    prev_lazy = None
    while prev_lazy != 0:
        g = {**data, **{"_root": data, "math": math}}
        data, cur_lazy = _resolve_once(data, g)
        if prev_lazy == cur_lazy:
            break
        prev_lazy = cur_lazy
    return data


def _cast(x, t):
    """Cast `x` to `t`, where `t` may be a Generic (for example Union or Tuple)."""

    # get_origin gets the generic "type", get_args the arguments in square
    # brackets. For example, get_origin(Union[str, int]) == Union and
    # get_args(Union[str, int]) == (str, int)
    orig = get_origin(t)
    if orig is None:
        return x if isinstance(t, type) and isinstance(x, t) else t(x)
    elif orig in (Union, Optional, UnionType):
        # Note: get_args(Optional[t]) == (t, None)
        # Check if any of the variant types matches the value exactly
        for ti in get_args(t):
            if isinstance(x, ti):
                return x
        # Otherwise, try to interpret the value as each of the variant types
        # and yield the first one to succeed.
        for ti in get_args(t):
            try:
                return _cast(x, ti)
            except:
                pass
        raise ValueError()
    elif orig in (Tuple, tuple):
        # In this case, since python 3.8, generics are the same as builtins
        # (i.e. Tuple == tuple), so we can use orig for construction
        return orig([_cast(xi, ti) for xi, ti in zip(x, get_args(t))])
    elif orig in (List, Set, list, set):
        ti = get_args(t)[0]
        return orig([_cast(xi, ti) for xi in x])
    elif orig in (Dict, dict):
        # Same story with dict
        kt, vt = get_args(t)
        return Namespace({k: _cast(xi, vt) for k, xi in x.items()})
    elif orig in (Literal,):
        # For Literals, check if the value is one of the allowed values.
        if x not in get_args(t):
            raise ValueError(x)
        return x
    else:
        raise NotImplementedError()


def _fix_types(x, t):
    """Align x with type t. If x is a dataclass, recursively fix types of its fields."""

    if hasattr(x, "__dataclass_fields__"):
        for name, field in x.__dataclass_fields__.items():
            value = getattr(x, name)
            value_t = field.type
            setattr(x, name, _fix_types(value, value_t))
        return x
    else:
        return _cast(x, t)


T = TypeVar("T")


def to_class(data: dict, cls: Type[T]) -> T:
    """Parse data into a class. The data may be obtained from e.g. YAML file, and cls must be a dataclass. One can use $(...) expressions for evaluation (like in Bash). Moreover, one can use non-builtin classes in the cls dataclass, which will get converted via the constructor from a base type."""

    res = dacite.from_dict(cls, data, dacite.Config(check_types=False))
    res = _fix_types(res, cls)

    return res


def update_(d1: dict, d2: dict):
    """Update `d1` with keys and values from `d2` in a level-wise fashion."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            update_(d1[k], v)
        else:
            d1[k] = v


def merge(*dicts: dict):
    """Merge one or more dicts into a single one."""
    merged = {}
    for d in dicts:
        update_(merged, d)
    return merged


def normalize(d: dict):
    """Normalize a dict, turning any key of form <k1>.<k2>...<kn>: v into k1: {k2: ... kn: v}."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = normalize(v)
        if "." in k:
            cur = res
            parts = [*k.split(".")]
            for ki in parts[:-1]:
                if ki not in cur:
                    cur[ki] = {}
                cur = cur[ki]
            k = parts[-1]
        else:
            cur = res

        if isinstance(v, dict):
            if k not in cur:
                cur[k] = {}
            update_(cur[k], v)
        else:
            cur[k] = v
    return res


def to_data(dicts: list[dict]) -> dict:
    """Get a merged, normalized and resolved config data dict from a number of
    "raw" dicts."""

    data = {}
    for dict in dicts:
        update_(data, normalize(dict))
        data = resolve_exprs(data)
    return data


def nested_index(d: dict, k: str):
    ki = k.split(".")
    cur = d
    for ki_ in ki[:-1]:
        cur = cur[ki_]
    return cur[ki[-1]]


def from_args(
    cls: Type[T] = None, defaults: Path = None, presets: Path = None
) -> T | dict:
    """Read config from command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="*")
    if presets is not None:
        parser.add_argument("-p", "--presets", nargs="*")

    args = parser.parse_args()

    dicts = []

    if defaults is not None:
        with open(defaults, "r") as f:
            data = yaml.safe_load(f) or {}
            dicts.append(data)

    if args.config is not None:
        with open(args.config, "r") as f:
            data = yaml.safe_load(f) or {}
            dicts.append(data)

    if presets is not None:
        if args.presets is not None:
            with open(presets, "r") as f:
                data = yaml.safe_load(f) or {}
                for preset in args.presets:
                    dicts.append(data.get(preset, {}))

    data = to_data(dicts)
    if cls is not None:
        data = to_class(data, cls)

    return data
