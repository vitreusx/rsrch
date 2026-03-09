from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import re
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar

import pyparsing as pp
from ruamel.yaml import YAML

from .cast import safe_partial

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


in_js_mode, in_upsert_mode, do_eval_templates = False, False, True


@contextmanager
def js_mode():
    """Enable accessing `Node` children via attr access, kinda like in JS.
    Used when evaluating templates, since we want `${key1.key2.key3}` to resolve to
    `key1["key2"]["key3"]`."""

    global in_js_mode  # noqa: PLW0603
    prev_mode = in_js_mode
    in_js_mode = True
    try:
        yield
    finally:
        in_js_mode = prev_mode


@contextmanager
def py_mode():
    """Disable accessing `Node` children via attr access, like in Python by default.
    When handling "Python-internal" stuff during template evaluation, we must
    disable access by attr access to avoid unexpected behavior."""

    global in_js_mode  # noqa: PLW0603
    prev_mode = in_js_mode
    in_js_mode = False
    try:
        yield
    finally:
        in_js_mode = prev_mode


@contextmanager
def upsert_mode(mode=True):
    """Automatically create a child `Node` on access, if it doesn't exist."""
    global in_upsert_mode  # noqa: PLW0603
    prev_mode = in_upsert_mode
    in_upsert_mode = mode
    try:
        yield
    finally:
        in_upsert_mode = prev_mode


@contextmanager
def eval_templates(mode=True):
    """Whether to eval templates (`${...}`'s)."""
    global do_eval_templates  # noqa: PLW0603
    prev_mode = do_eval_templates
    do_eval_templates = mode
    try:
        yield
    finally:
        do_eval_templates = prev_mode


class NodeLocals:
    """A quasi-`locals()` mapping for config nodes."""

    def __init__(self, node: Node):
        self.node = node
        # self._pydevd is necessary for python debugger to work
        self._pydevd = {}

    def up(self):
        return NodeLocals(self.node.parent)

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


locator = pp.Empty().setParseAction(lambda s, l_, t: l_)  # noqa: ARG005


def locatedExpr(expr):  # noqa: N802
    return pp.Group(locator("start") + expr("value") + locator("end"))


class TemplateEngine:
    """An engine for evaluating `${...}` items."""

    EXPR = locatedExpr(pp.nestedExpr("${", "}"))
    EVAL_RE = r"^((?P<resolver>[\w]+):)?(?P<expr>.*)$"
    VAR_RE = r"^((?P<up>\.*)(?P<var>[a-zA-Z0-9_\.]+))$"

    @classmethod
    def render(cls, text: str, locals: NodeLocals):
        exprs = []
        for m in cls.EXPR.searchString(text).asList():
            beg, _, end = m[0]
            exprs.append((beg, end))

        if len(exprs) == 1 and exprs[0] == (0, len(text)):
            return cls._eval(text[2:-1], locals)

        cur, res = 0, []
        for beg, end in exprs:
            res.append(text[cur:beg])
            eval_r = cls._eval(text[beg + 2 : end - 1], locals)
            if not isinstance(eval_r, str):
                eval_r = str(eval_r)
            res.append(eval_r)
            cur = end
        res.append(text[cur:])
        return "".join(res)

    @classmethod
    def _eval(cls, expr: str, locals: NodeLocals):
        m = re.match(cls.VAR_RE, expr)
        if m is not None:
            up_count = max(len(m["up"]) - 1, 0)
            for _ in range(up_count):
                locals = locals.up()
            with js_mode():
                return eval(m["var"], None, locals)  # noqa: S307

        m = re.match(cls.EVAL_RE, expr)
        if m is not None:
            resolver = m["resolver"] or "eval"
            if resolver == "eval":
                with js_mode():
                    return eval(m["expr"], None, locals)  # noqa: S307
            elif resolver == "env":
                return os.environ[m["expr"]]
            else:
                raise ValueError("Unsupported resolver '%s'", resolver)

        raise ValueError("String '%s' is not a valid expression.", expr)


class Node(MutableMapping):
    """A config node.

    Represents a YAML node (dict, list or scalar). Due to the presence of
    templates, we can't just use native Python classes, for example due to
    having to keep track of "variables" accessible from a given node, and having
    to automatically evaluate templates on access, if neccessary.

    The `value` passed to the constructor is modified in-place.
    """

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
            if (
                in_upsert_mode
                and isinstance(self.value, dict)
                and key not in self.value
            ):
                self[key] = {}

            value = self.value[key]

            if isinstance(value, str) and do_eval_templates:
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
        return TemplateEngine.render(value, NodeLocals(self))

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
        else:
            raise TypeError(type(self.value))

    def __repr__(self):
        return f"node({self.value!r})"


def _render_all(value):
    if isinstance(value, Node):
        unwrapped = value.value
        if isinstance(unwrapped, dict):
            return {k: _render_all(v) for k, v in value.items()}
        elif isinstance(unwrapped, list):
            return [_render_all(elem) for elem in value]
        else:
            raise TypeError(type(value))
    else:
        return value


def render_all(cfg):
    cfg = Node(cfg)
    cfg = _render_all(cfg)
    return cfg


def _merge(base, other):
    if not (
        isinstance(base, Node)
        and isinstance(base.value, dict)
        and isinstance(other, Node)
        and isinstance(other.value, dict)
    ):
        return other

    if other.get("$replace", False):
        return other

    with eval_templates(False):
        for key in other:
            if key.startswith("$"):
                continue
            with js_mode():
                with upsert_mode():
                    exec(f"base.{key}")  # noqa: S102
                exec(f"base.{key} = _merge(base.{key}, value)")  # noqa: S102

    return base


def _apply_preset(base: Node, preset: Node):
    if "$extends" in preset:
        extends: str | list[str] = preset["$extends"]
        if isinstance(extends, str):
            extends = [extends]
        for ext_name in extends:
            ext = preset.render("${" + ext_name + "}")
            _apply_preset(base, ext)
    _merge(base, preset)


def apply_preset(base: dict, preset: dict):
    _apply_preset(Node(base), Node(preset))


def apply_presets(base: dict, all_presets: dict, presets: list[str]):
    """Apply a set of presets to a config object."""

    base, all_presets = Node(base), Node(all_presets)

    for name in presets:
        with js_mode():
            preset: Node = eval(f"all_presets.{name}")  # noqa: S307
        _apply_preset(base, preset)


def hide_private(x):
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(k, str) and (k.startswith(("_", "$"))):
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
    presets_dir: str | Path | None = None,
):
    if (presets_yml is not None) and (presets_dir is not None):
        raise ValueError("Only one of `presets_yml`, `presets_dir` can be provided.")

    p = argparse.ArgumentParser()
    p.add_argument(
        "-C",
        "--config-file",
        type=Path,
        default=Path(config_yml),
        help="Path to config.yml file with default config values.",
    )
    p.add_argument(
        "-P",
        "--preset-files",
        type=Path,
        nargs="*",
        default=[],
        help="Extra preset files to use.",
    )
    p.add_argument(
        "-o",
        "--options",
        help="Manual options, in the form of a preset.",
    )
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Dump the config to stdout and exit.",
    )
    p.add_argument(
        "-p",
        "--presets",
        type=str,
        nargs="+",
        default=[],
        help="List of presets to be used.",
    )

    args = p.parse_args()

    cfg = open(args.config_file)

    preset_files = []
    if presets_yml is not None:
        preset_files = [presets_yml]
    elif presets_dir is not None:
        preset_files = [*Path(presets_dir).iterdir()]

    if len(args.preset_files):
        preset_files.extend(args.preset_files)

    all_presets = {}
    for preset_file in preset_files:
        all_presets.update(open(preset_file))

    if all_presets is not None:
        apply_presets(cfg, all_presets, args.presets)

    if args.options is not None:
        apply_preset(cfg, load(args.options))

    cfg = eval_vars(cfg)

    if args.dump_config:
        json.dump(cfg, sys.stdout)
        sys.exit(0)

    return cfg


T = TypeVar("T")


class Dynamic(Generic[T]):
    """A config type for "dynamically typed" objects.

    Typical use case is as follows: when you design a config file for your
    training procedure, and want to leave e.g. backbone or optimizer choice
    completely up to the user, you can add them as `Dynamic` objects. A following
    example YAML config:

    ```
    optimizer:
      $class: torch.optim.AdamW
      lr: 3e-4
      eps: 1e-5
    ```

    is converted to `optimizer: Dynamic`, and `optimizer.create()` returns an
    instance of `torch.optim.AdamW`.

    One can use a generic annotation (`Dynamic[T]`) to provide a hint that
    the constructed value is of type `T`.
    """

    def __init__(self, **kwargs):
        cls: str = kwargs["$class"]
        index = cls.rfind(".")
        if index < 0:
            raise RuntimeError("$class value `%s` needs to be fully qualified.", cls)
        module = importlib.import_module(cls[:index])
        self.cls: type = getattr(module, cls[index + 1 :])
        del kwargs["$class"]
        self._ctor = safe_partial(self.cls, **kwargs)

    def create(self, *args, **kwargs) -> T:
        return self._ctor(*args, **kwargs)
