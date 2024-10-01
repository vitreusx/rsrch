from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

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


in_js_mode, in_upsert_mode = False, False


@contextmanager
def js_mode():
    """Enable accessing mapping items via attr access, kinda like JS."""
    global in_js_mode
    prev_mode = in_js_mode
    in_js_mode = True
    try:
        yield
    finally:
        in_js_mode = prev_mode


@contextmanager
def py_mode():
    """Disable accessing mapping items via attr access."""
    global in_js_mode
    prev_mode = in_js_mode
    in_js_mode = False
    try:
        yield
    finally:
        in_js_mode = prev_mode


@contextmanager
def upsert_mode(mode=True):
    """Automatically create nodes on access, if not existent."""
    global in_upsert_mode
    prev_mode = in_upsert_mode
    in_upsert_mode = mode
    try:
        yield
    finally:
        in_upsert_mode = prev_mode


locator = pp.Empty().setParseAction(lambda s, l, t: l)


def locatedExpr(expr):
    return pp.Group(locator("start") + expr("value") + locator("end"))


class Locals:
    def __init__(self, node: "Node"):
        self.node = node
        self._pydevd = {}

    def up(self):
        return Locals(self.node.parent)

    def __getitem__(self, var_name: str):
        if var_name in self._pydevd:
            return self._pydevd[var_name]

        with py_mode():
            cur = self.node
            while cur is not None:
                if var_name in cur:
                    return cur[var_name]
                cur = cur.parent

        raise KeyError(var_name)

    def __setitem__(self, name: str, value: Any):
        self._pydevd[name] = value


class Renderer:
    EXPR = locatedExpr(pp.nestedExpr("${", "}"))
    EVAL_RE = r"^((?P<resolver>[\w]+):)?(?P<expr>.*)$"
    VAR_RE = r"^((?P<up>\.*)(?P<var>[a-zA-Z0-9_\.]+))$"

    @classmethod
    def render(self, text: str, locals: Locals):
        exprs = []
        for m in self.EXPR.searchString(text).asList():
            beg, _, end = m[0]
            exprs.append((beg, end))

        if len(exprs) == 1 and exprs[0] == (0, len(text)):
            return self.eval(text[2:-1], locals)

        cur, res = 0, []
        for beg, end in exprs:
            res.append(text[cur:beg])
            eval_r = self.eval(text[beg + 2 : end - 1], locals)
            if not isinstance(eval_r, str):
                eval_r = str(eval_r)
            res.append(eval_r)
            cur = end
        res.append(text[cur:])
        return "".join(res)

    @classmethod
    def eval(self, expr: str, locals: Locals):
        m = re.match(self.VAR_RE, expr)
        if m is not None:
            up_count = max(len(m["up"]) - 1, 0)
            for _ in range(up_count):
                locals = locals.up()
            with js_mode():
                return eval(m["var"], None, locals)

        m = re.match(self.EVAL_RE, expr)
        if m is not None:
            resolver = m["resolver"] or "eval"
            if resolver == "eval":
                with js_mode():
                    return eval(m["expr"], None, locals)
            elif resolver == "env":
                return os.environ[m["expr"]]
            else:
                raise ValueError(f"Unsupported resolver '{resolver}'")

        raise ValueError(f"String '{expr}' is not a valid expression.")


class Node(MutableMapping):
    def __init__(self, value: Any, parent: Node | None = None):
        self.value = value
        self.parent = parent

    def __getattr__(self, name: str):
        if in_js_mode:
            return self[name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any):
        if in_js_mode:
            self[name] = value
        else:
            super().__setattr__(name, value)

    def __getitem__(self, key):
        with py_mode():
            if in_upsert_mode:
                if isinstance(self.value, dict):
                    if key not in self.value:
                        self[key] = {}

            value = self.value[key]

            if isinstance(value, str):
                value = self.render(value)

            if isinstance(value, (list, dict)):
                value = Node(value, self)

            return value

    def __contains__(self, key):
        if isinstance(self.value, (dict, list)):
            return key in self.value
        else:
            return False

    def render(self, value):
        return Renderer.render(value, Locals(self))

    def __setitem__(self, key, value):
        with py_mode():
            if isinstance(value, Node):
                value = value.value
            self.value[key] = value

    def __delitem__(self, key):
        with py_mode():
            del self.value[key]

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        if isinstance(self.value, dict):
            return iter(self.value)
        elif isinstance(self.value, list):
            return (self[i] for i in range(len(self)))

    def __repr__(self):
        return f"node({self.value!r})"


def _merge(base, other):
    if not (
        isinstance(base, Node)
        and isinstance(base.value, dict)
        and isinstance(other, Node)
        and isinstance(other.value, dict)
    ):
        return other

    if "$replace" in other:
        return other

    for key, value in other.items():
        if key.startswith("$"):
            continue
        with js_mode():
            with upsert_mode():
                exec(f"base.{key}")
            exec(f"base.{key} = _merge(base.{key}, value)")

    return base


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


def apply_preset(base, preset):
    _merge(Node(base), Node(preset))


def apply_presets(base: dict, all_presets: dict, presets: list[str]):
    base, all_presets = Node(base), Node(all_presets)

    for name in presets:
        with js_mode():
            preset: Node = eval(f"all_presets.{name}")

        assert isinstance(preset.value, dict)
        if "$extends" in preset:
            extends = preset["$extends"]
            if isinstance(extends, str):
                extends = [extends]
            for ext_name in extends:
                ext = preset.render("${" + ext_name + "}")
                _merge(base, ext)

        _merge(base, preset)


def hide_private(x):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, str) and (k.startswith("_") or k.startswith("$")):
                continue
            r[k] = hide_private(v)
        return r
    elif isinstance(x, list):
        return [hide_private(xi) for xi in x]
    else:
        return x


T = TypeVar("T")


def eval_vars(data: dict):
    data = render_all(data)
    data = hide_private(data)
    return data


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
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Dump the config to stdout and exit.",
    )

    args = p.parse_args()

    cfg = open(args.config_file)

    if presets_yml is not None:
        all_presets = open(args.presets_file)
        apply_presets(cfg, all_presets, args.presets)

    if args.options is not None:
        apply_preset(cfg, load(args.options))

    cfg = eval_vars(cfg)

    if args.dump_config:
        yaml.dump(cfg, sys.stdout)
        exit(0)

    return cfg
