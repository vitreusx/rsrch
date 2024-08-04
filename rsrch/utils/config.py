"""Config parsing library."""

import json
import math
import os
import re
import types
import typing
from argparse import Namespace
from collections import UserDict
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Type, TypeVar, get_args, get_origin

import numpy as np
import pyparsing as pp
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


def to_attr_dict(x: Any):
    if isinstance(x, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in x.items()})
    elif isinstance(x, list):
        return [to_attr_dict(v) for v in x]
    else:
        return x


locator = pp.Empty().setParseAction(lambda s, l, t: l)


def locatedExpr(expr):
    return pp.Group(locator("start") + expr("value") + locator("end"))


class Resolver:
    EXPR = locatedExpr(pp.nested_expr("${", "}"))

    VAR_RE = r"(?P<up>\.+)(?P<var>[\w\.]+)"
    EVAL_RE = r"((?P<resolver>[\w]+):)?(?P<expr>.*)"

    def resolve(self, value: Any):
        while True:
            self.stack = [value]
            value, is_same = self._single_pass(value)
            if is_same:
                break
        return value

    def _single_pass(self, x: Any):
        is_same = True
        if isinstance(x, dict):
            new_x = {}
            for k, v in x.items():
                self.stack.append(v)
                v, is_v_same = self._single_pass(v)
                self.stack.pop()
                is_same &= is_v_same
                new_x[k] = v
            x = new_x
        elif isinstance(x, list):
            new_x = []
            for i, v in enumerate(x):
                self.stack.append(v)
                v, is_v_same = self._single_pass(v)
                self.stack.pop()
                is_same &= is_v_same
                new_x.append(v)
            x = new_x
        elif isinstance(x, str):
            new_x = self._eval_text(x)
            is_same &= x == new_x
            x = new_x
        return x, is_same

    def _eval_text(self, text: str):
        exprs = []
        for m in self.EXPR.search_string(text).as_list():
            beg, _, end = m[0]
            exprs.append((beg, end))

        if len(exprs) == 1 and exprs[0] == (0, len(text)):
            return self._eval_expr(text[2:-1])
        else:
            cur, res = 0, []
            for beg, end in exprs:
                res.append(text[cur:beg])
                eval_r = self._eval_expr(text[beg + 2 : end - 1])
                if not isinstance(eval_r, str):
                    eval_r = str(eval_r)
                res.append(eval_r)
                cur = end
            res.append(text[cur:])
            return "".join(res)

    def _eval_expr(self, expr: str):
        m = re.match(self.VAR_RE, expr)
        if m is not None:
            backtrack, name = len(m["up"]), m["var"]
            value = self.stack[-backtrack - 1]
            for part in name.split("."):
                value = value[part]
            return value

        m = re.match(self.EVAL_RE, expr)
        if m is not None:
            resolver = m["resolver"] or "eval"
            if resolver == "eval":
                g = {"math": math, "np": np}
                for scope in self.stack[:-1]:
                    if isinstance(scope, dict):
                        g.update(scope)
                g = to_attr_dict(g)
                return eval(m["expr"], g)
            elif resolver == "env":
                return os.environ[m["expr"]]
            else:
                raise ValueError(f"Unknown resolver '{resolver}'")

        raise ValueError(f"Could not interpret expression '{expr}'.")


def cast(x, t):
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
                # For some insane reason, sometimes field.type is a name
                # of the class, like 'str' or 'bool'
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


T = TypeVar("T")


def hide_private(x: Any):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            r[k] = hide_private(v)
        return r
    elif isinstance(x, list):
        return [hide_private(xi) for xi in x]
    else:
        return x


def unravel(x: Any):
    """Recursively replace all keys in the dict of form <k1>.<k2>...: v
    with k1: {k2: ... }."""

    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            parts = k.split(".")
            cur = r
            for part in parts[:-1]:
                if part not in cur:
                    cur[part] = {}
                cur = cur[part]
            cur[parts[-1]] = unravel(v)
    elif isinstance(x, list):
        r = [unravel(xi) for xi in x]
    else:
        r = x
    return r


def ravel(x: Any):
    r = {}

    def walk(cur, dest=None):
        if isinstance(cur, dict):
            for k, v in cur.items():
                walk(v, k if dest is None else f"{dest}.{k}")
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                walk(v, str(i) if dest is None else f"{dest}.{i}")
        else:
            r[dest] = cur

    walk(x)
    return r


def parse(data: dict, cls: Type[T]) -> T:
    """Given a config dict, parse it into a config object of a given type."""
    data = unravel(data)
    data = Resolver().resolve(data)
    data = hide_private(data)
    return cast(data, cls)


def merge_(dst: Any, src: Any):
    src = unravel(src)

    if not isinstance(src, dict) or not isinstance(dst, dict):
        return src

    for k, v in src.items():
        cur = dst
        k_parts = k.split(".")
        for k_part in k_parts[:-1]:
            if not isinstance(cur.get(k_part), dict):
                cur[k_part] = {}
            cur = cur[k_part]
        cur[k_parts[-1]] = merge_(cur.get(k_parts[-1]), v)
    return dst


def merge(*dicts: dict):
    r = {}
    for d in dicts:
        r = merge_(r, d)
    return r


def load(path: str | Path):
    """Load data from a .json or .yml file."""

    path = Path(path)
    if path.suffix in (".yml", ".yaml"):
        with open(path, "r") as f:
            return yaml.load(f)
    elif path.suffix in (".json",):
        with open(path, "r") as f:
            return json.load(f)


def add_preset_(cfg: dict, presets: dict, name: str):
    """Update config dict with a preset. If the preset contains `$extends` key,
    the indicated presets are added as well."""

    preset = presets.get(name, {})
    if "$extends" in preset:
        for ext in preset["$extends"]:
            add_preset_(cfg, presets, ext)
        del preset["$extends"]
    merge_(cfg, preset)
