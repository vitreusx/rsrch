import argparse
import io
import math
import os
import re
import types
import typing
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, Type, TypeVar, get_args, get_origin

import numpy as np
import pyparsing as pp
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)


_open = open


def open(path: str | Path):
    """Open a YAML file."""
    with _open(path, "r") as f:
        return yaml.load(f)


def load(content: str):
    """Parse a string into a YAML object."""
    buf = io.StringIO(content)
    return yaml.load(buf)


locator = pp.Empty().setParseAction(lambda s, l, t: l)


def locatedExpr(expr):
    return pp.Group(locator("start") + expr("value") + locator("end"))


_items_as_attrs = False
"""Whether attribute access for `Node` objects should be treated as item access."""


@contextmanager
def enable_items_as_attrs():
    """Enable accessing mapping items via attr access."""
    global _items_as_attrs
    prev = _items_as_attrs
    _items_as_attrs = True
    try:
        yield
    finally:
        _items_as_attrs = prev


@contextmanager
def disable_items_as_attrs():
    """Disable accessing mapping items via attr access."""
    global _items_as_attrs
    prev = _items_as_attrs
    _items_as_attrs = False
    try:
        yield
    finally:
        _items_as_attrs = prev


class Template:
    """A string template.

    A template is a string with expressions, denoted by opening `${` and closing `}`. These expressions can be replaced by their values using `render` function.
    """

    EXPR = locatedExpr(pp.nested_expr("${", "}"))
    EVAL_RE = r"((?P<resolver>[\w]+):)?(?P<expr>.*)"

    def __init__(self, text: str):
        self.text = text

    def render(self, locals={}):
        """Render a template, given a locals mapping."""

        exprs = []
        for m in self.EXPR.search_string(self.text).as_list():
            beg, _, end = m[0]
            exprs.append((beg, end))

        if len(exprs) == 1:
            return self._eval_expr(self.text[2:-1], locals)

        cur, res = 0, []
        for beg, end in exprs:
            res.append(self.text[cur:beg])
            eval_r = self._eval_expr(self.text[beg + 2 : end - 1], locals)
            if not isinstance(eval_r, str):
                eval_r = str(eval_r)
            res.append(eval_r)
            cur = end
        res.append(self.text[cur:])
        return "".join(res)

    def _eval_expr(self, expr: str, locals={}):
        m = re.match(self.EVAL_RE, expr)
        if m is None:
            raise ValueError(f"String '{expr}' is not a valid expression.")

        resolver = m["resolver"] or "eval"
        if resolver == "eval":
            locals_ = {"math": math, "np": np, **locals}
            with enable_items_as_attrs():
                return eval(m["expr"], None, locals_)
        elif resolver == "env":
            return os.environ[m["expr"]]
        else:
            raise ValueError(f"Unknown resolver '{resolver}'")

    def __repr__(self):
        return repr(self.text)


class Node(MutableMapping):
    """A YAML node.

    A node behaves much like a list or a dict, except:
    - One can access the subnodes via attribute access.
    - All the templates are automatically rendered.
    """

    def __init__(self, value, parent=None):
        if isinstance(value, Node):
            value = value.unwrapped
        self.unwrapped = value
        self.parent = parent

    @property
    def scope(self):
        stack, cur = [], self
        while cur is not None:
            if isinstance(cur.unwrapped, dict):
                stack.append(cur.unwrapped)
            cur = cur.parent
        stack.reverse()
        scope = {k: v for scope in stack for k, v in scope.items()}
        for k, v in scope.items():
            if isinstance(v, (dict, list)):
                scope[k] = Node(v)
        return scope

    def __getattr__(self, name: str):
        if _items_as_attrs and not name.startswith("__"):
            return self[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value):
        if _items_as_attrs and not name.startswith("__"):
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str):
        if _items_as_attrs and not name.startswith("__"):
            del self[name]
        else:
            super().__delattr__(name)

    def __getitem__(self, key):
        with disable_items_as_attrs():
            if isinstance(self.unwrapped, dict):
                if key not in self:
                    self[key] = {}

            child = self.unwrapped[key]

            if isinstance(child, str):
                child = Template(child).render(self.scope)

            if isinstance(child, (list, dict)):
                child = Node(child, parent=self)

            return child

    def __contains__(self, key):
        return key in self.unwrapped

    def __setitem__(self, key, value):
        with disable_items_as_attrs():
            if isinstance(value, Node):
                value = value.unwrapped
            self.unwrapped[key] = value

    def __delitem__(self, key):
        del self.unwrapped[key]

    def __len__(self):
        return len(self.unwrapped)

    def __iter__(self):
        if isinstance(self.unwrapped, dict):
            return iter(self.unwrapped)
        else:
            return (self[i] for i in range(len(self)))

    def __repr__(self):
        return repr(self.unwrapped)


def _render_all(value):
    if isinstance(value, Node):
        unwrapped = value.unwrapped
        if isinstance(unwrapped, dict):
            return {k: _render_all(v) for k, v in value.items()}
        elif isinstance(unwrapped, list):
            return [_render_all(elem) for elem in value]
    else:
        return value


def render_all(cfg):
    cfg = Node(cfg)
    cfg = _render_all(cfg)
    return cfg


def _use_preset(node_b: Node, preset, node_p: Node):
    if not isinstance(preset, dict):
        return preset

    for k, v in preset.items():
        with disable_items_as_attrs():
            k2 = Template(k).render({**node_b.scope, **node_p.scope})
        with enable_items_as_attrs():
            exec(f"node_b.{k2} = _use_preset(node_b.{k2}, v, node_p[k])")

    return node_b


def use_preset(base: dict, preset: dict):
    node_b, node_p = Node(base), Node(preset)
    _use_preset(node_b, preset, node_p)


def compose(base: dict, *presets: dict):
    base = {**base}
    for preset in presets:
        use_preset(base, preset)
    base = render_all(base)
    base = _hide_private(base)
    return base


def _cast(x: Any, t: Type):
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
            args[field.name] = _cast(x[field.name], field_t)

        return t(**args)
    elif t in (typing.Union, typing.Optional, types.UnionType):
        for ti in t_args:
            ti_ = get_origin(ti) or ti
            if isinstance(ti_, type) and isinstance(x, ti_):
                return x
        for ti in t_args:
            try:
                return _cast(x, ti)
            except:
                pass
        raise ValueError(f"None of the variant types {t_args} match value {x}")
    elif t in (typing.Tuple, tuple):
        return tuple([_cast(xi, ti) for xi, ti in zip(x, t_args)])
    elif t in (typing.List, typing.Set, list, set):
        elem_t = t_args[0]
        return t([_cast(xi, elem_t) for xi in x])
    elif t in (typing.Dict, dict):
        if len(t_args) > 0:
            kt, vt = t_args
        else:
            kt, vt = Any, Any
        return {_cast(k, kt): _cast(xi, vt) for k, xi in x.items()}
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


def _hide_private(x: Any):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            r[k] = _hide_private(v)
        return r
    elif isinstance(x, list):
        return [_hide_private(xi) for xi in x]
    else:
        return x


T = TypeVar("T")


def cast(cfg: dict, typ: Type[T]) -> T:
    cfg = _hide_private(cfg)
    return _cast(cfg, typ)


def cli(
    config_yml: str | Path,
    presets_yml: str | Path | None = None,
):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=Path(config_yml),
        help="Path to config.yml file with default config values.",
    )
    if presets_yml is not None:
        p.add_argument(
            "-P",
            "--presets-file",
            type=Path,
            default=Path(presets_yml),
            help="Path to presets.yml file with available presets.",
        )
        p.add_argument(
            "-p",
            "--presets",
            type=str,
            nargs="+",
            help="List of presets to be used.",
        )

    args, unk = p.parse_known_args()

    base = open(args.config_file)

    presets = []
    if presets_yml is not None:
        all_presets = open(args.presets_file)
        for name in args.presets:
            preset = eval(f"Node(all_presets).{name}")
            presets.append(preset)

    idx = 0
    while idx < len(unk):
        val = unk[idx]
        if val.startswith("--"):
            key = val.removeprefix("--")
            stop = idx + 1
            while stop < len(unk) and not unk[stop].startswith("--"):
                stop += 1
            vals = val[idx + 1 : stop]
            if len(vals) == 1:
                vals = vals[0]
            presets.append({key: vals})
            idx = stop
        else:
            idx += 1

    return compose(base, *presets)


__all__ = ["compose", "cast", "cli"]
