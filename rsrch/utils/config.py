import argparse
import io
import math
import os
import re
from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TypeVar

import numpy as np
import pyparsing as pp
from ruamel.yaml import YAML

from .cast import cast

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
def node_item_mode():
    """Enable accessing mapping items via attr access."""
    global _items_as_attrs
    prev = _items_as_attrs
    _items_as_attrs = True
    try:
        yield
    finally:
        _items_as_attrs = prev


@contextmanager
def node_attr_mode():
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

    EXPR = locatedExpr(pp.nestedExpr("${", "}"))
    EVAL_RE = r"((?P<resolver>[\w]+):)?(?P<expr>.*)"

    def __init__(self, text: str):
        self.text = text

    def render(self, locals={}):
        """Render a template, given a locals mapping."""

        exprs = []
        for m in self.EXPR.searchString(self.text).asList():
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
            with node_item_mode():
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
    - One can access the subnodes via attribute access, if within `node_attr_mode` context.
    - All the templates are automatically rendered.
    """

    def __init__(self, value, parent=None):
        if isinstance(value, Node):
            value = value.value
        self.value = value
        self.parent = parent

    @property
    def scope(self):
        stack, cur = [], self
        while cur is not None:
            if isinstance(cur.value, dict):
                stack.append(cur.value)
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
        with node_attr_mode():
            if isinstance(self.value, dict):
                if key not in self:
                    self[key] = {}

            child = self.value[key]

            if isinstance(child, str):
                child = Template(child).render(self.scope)

            if isinstance(child, (list, dict)):
                child = Node(child, parent=self)

            return child

    def __contains__(self, key):
        return key in self.value

    def __setitem__(self, key, value):
        with node_attr_mode():
            if isinstance(value, Node):
                value = value.value
            self.value[key] = value

    def __delitem__(self, key):
        del self.value[key]

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        if isinstance(self.value, dict):
            return iter(self.value)
        else:
            return (self[i] for i in range(len(self)))

    def __repr__(self):
        return repr(self.value)


def _render_all(value):
    if isinstance(value, Node):
        unwrapped = value.value
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


def _apply_preset(base, preset):
    if not (
        isinstance(base, Node)
        and isinstance(base.value, dict)
        and isinstance(preset, Node)
        and isinstance(preset.value, dict)
    ):
        return preset

    if "$replace" in preset:
        return preset

    for key, value in preset.items():
        if key.startswith("$"):
            continue
        with node_item_mode():
            exec(f"base.{key} = _apply_preset(base.{key}, value)")

    return base


def apply_preset(base, preset):
    return _apply_preset(Node(base), Node(preset))


def apply_presets(base: dict, all_presets: dict, presets: list[str]):
    base, all_presets = Node(base), Node(all_presets)

    for name in presets:
        with node_item_mode():
            preset: Node = eval(f"all_presets.{name}")

        assert isinstance(preset.value, dict)
        if "$extends" in preset:
            extends = preset["$extends"]
            if not isinstance(extends, list):
                extends = [extends]
            for ext_name in extends:
                ext: Node = Template(f"${{{ext_name}}}").render(preset.scope)
                _apply_preset(base, ext)

        _apply_preset(base, preset)


def _hide_private(x):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, str) and (k.startswith("_") or k.startswith("$")):
                continue
            r[k] = _hide_private(v)
        return r
    elif isinstance(x, list):
        return [_hide_private(xi) for xi in x]
    else:
        return x


T = TypeVar("T")


def cli(
    config_yml: str | Path,
    presets_yml: str | Path | None = None,
    def_presets: list[str] | None = [],
):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=Path(config_yml),
        help="Path to config.yml file with default config values.",
    )
    p.add_argument(
        "-o",
        "--options",
        help="Manual options, in the form of a preset.",
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
            default=def_presets,
            help="List of presets to be used.",
        )

    args = p.parse_args()

    base = open(args.config_file)

    if presets_yml is not None:
        all_presets = open(args.presets_file)
        apply_presets(base, all_presets, args.presets)

    if args.options is not None:
        apply_preset(base, load(args.options))

    base = render_all(base)
    base = _hide_private(base)
    return base


__all__ = ["compose", "cast", "cli"]
