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
        with node_attr_mode():
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
        with node_attr_mode():
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


def _use_preset(node_b, preset, node_p):
    with node_attr_mode():
        if not (
            isinstance(node_b, Node)
            and isinstance(node_b.unwrapped, dict)
            and isinstance(node_p, Node)
            and isinstance(node_p.unwrapped, dict)
            and isinstance(preset, dict)
        ):
            return preset

        if preset.get("$replace"):
            # Usually, we recursively merge dicts. Sometimes, however, it's necessary to replace the target node entirely. We can indicate this via `$replace` key.
            preset = {**preset}
            del preset["$replace"]
            return preset

        for k, v in preset.items():
            k2 = Template(k).render({**node_b.scope, **node_p.scope})
            with node_item_mode():
                exec(f"node_b.{k2} = _use_preset(node_b.{k2}, v, node_p[k])")

    return node_b


def use_preset(base: dict, preset):
    node_b, node_p = Node(base), Node(preset)
    _use_preset(node_b, preset, node_p)


def compose(base: dict, presets: list[dict]):
    base = {**base}
    for preset in presets:
        use_preset(base, preset)
    base = render_all(base)
    base = _hide_private(base)
    return base


def find_by_path(d, key):
    node = Node(d)
    with node_item_mode():
        r = eval(f"node.{key}")
    if isinstance(r, Node):
        r = r.unwrapped
    return r


def get_presets(all_presets: dict, names: list[str]):
    r = []
    for name in names:
        preset = find_by_path(all_presets, name)
        if "$extends" in preset:
            r.extend(get_presets(all_presets, preset["$extends"]))
            del preset["$extends"]
        r.append(preset)
    return r


def _hide_private(x):
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

    presets = []
    if presets_yml is not None:
        all_presets = open(args.presets_file)
        presets.extend(get_presets(all_presets, args.presets))

    if args.options is not None:
        options = load(args.options)
        presets.append(options)

    return compose(base, presets)


__all__ = ["compose", "cast", "cli"]
