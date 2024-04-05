import argparse
import ast
import inspect
import io
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

import numpy as np
from mako.template import Template
from ruamel.yaml import YAML

yaml = YAML(typ="safe", pure=True)

__all__ = ["from_dicts", "cli", "parser", "from_args", "parse", "dataclass"]


class AttrDict(dict):
    def __getattr__(self, __name):
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return self[__name]


class Expr:
    """An expression object."""

    VAR_PAT = r"^\${(?P<back>\.*)(?P<var>[\w\.]+)}"  # e.g. ${..opt.lr}
    EXPR_PAT = r"^\${(?P<expr>.*)}$"  # e.g. $(-1/np.log(2))

    def __init__(self, value: str, path: list):
        self._value = value
        self._path = path
        self.eval = self._make_eval_fn()

    def _make_eval_fn(self):
        m = re.match(self.VAR_PAT, self._value)
        if m is not None:
            return self._eval_var(m)

        m = re.match(self.EXPR_PAT, self._value)
        if m is not None:
            return self._eval_expr(m)

        return self._eval_mako

    def _eval_var(self, m: re.Match):
        def _fn(g: dict):
            eval_g = g
            if len(m["back"]) > 0:
                cur, local = g, g
                pivot = len(self._path) - len(m["back"])
                for k in self._path[:pivot]:
                    cur = cur[k]
                    if isinstance(cur, dict):
                        local = cur
                eval_g = local

            return eval(m["var"], eval_g)

        return _fn

    def _eval_expr(self, m: re.Match):
        def _fn(g: dict):
            cur, local = g, g
            for k in self._path[:-1]:
                cur = cur[k]
                if isinstance(cur, dict):
                    local = cur

            modules = {"np": np, "math": math}
            g = {**g, **modules, **local}
            return eval(m["expr"], g)

        return _fn

    def _eval_mako(self, g: dict):
        cur, local = g, g
        for k in self._path[:-1]:
            cur = cur[k]
            if isinstance(cur, dict):
                local = cur

        modules = {"np": np, "math": math}
        g = {**g, **modules, **local}
        if "buffer" in g:
            # Due to a limitation in Mako, we cannot pass "buffer" variable
            # directly, if it is defined.
            assert "_buffer" not in g
            g["_buffer"] = g["buffer"]
            del g["buffer"]

        return Template(self._value).render(**g)

    def __repr__(self):
        loc = ".".join(str(x) for x in self._path)
        return f"Expr({loc}: {self._value})"


def replace_str_by_expr(value: Any, path=[]):
    """Replace string fields in the object by Expr instances."""
    if isinstance(value, dict):
        new_value = {}
        for k, v in [*value.items()]:
            new_value[k] = replace_str_by_expr(v, [*path, k])
        value = new_value
    elif isinstance(value, list):
        for i, v in enumerate(value):
            value[i] = replace_str_by_expr(v, [*path, i])
    elif isinstance(value, str):
        value = Expr(value, path)
    return value


def resolve_exprs(x: Any):
    """Convert Expr objects to regular values via evaluation, recursively (that is, expressions can refer to each other.)"""

    def _resolve_once(value: Any, g: dict):
        """Convert Expr objects to regular values via evaluation, non-recursively."""
        changed = False
        if isinstance(value, dict):
            for k, v in [*value.items()]:
                new_v, v_changed = _resolve_once(v, g)
                value[k] = new_v
                changed |= v_changed
        elif isinstance(value, list):
            for i, v in enumerate(value):
                new_v, v_changed = _resolve_once(v, g)
                value[i] = new_v
                changed |= v_changed
        elif isinstance(value, Expr):
            try:
                value = value.eval(g)
                changed |= not isinstance(value, Expr)
            except:
                ...

        return value, changed

    def _g(x):
        d = {}
        for k, v in x.items():
            d[k] = _g(v) if isinstance(v, dict) else v
            # if not isinstance(v, Expr):
            #     d[k] = _g(v) if isinstance(v, dict) else v
        return AttrDict(d)

    while True:
        x, changed = _resolve_once(x, _g(x))
        if not changed:
            break

    return x


def cast(x, t):
    """Cast `x` to `t`, where `t` may be a Generic (for example Union or Tuple). Raise ValueError, if the value cannot be converted to the specified type."""

    # get_origin gets the generic "type", get_args the arguments in square
    # brackets. For example, get_origin(Union[str, int]) == Union and
    # get_args(Union[str, int]) == (str, int)
    orig = get_origin(t)
    if t is None or t == type(None):
        if x is not None:
            raise ValueError()
        return x
    elif is_dataclass(t):
        # Allow conversion of dicts to dataclasses
        args = {}
        for field in fields(t):
            if field.name in x:
                field_t = field.type
                if isinstance(field_t, str):
                    field_t = eval(field_t)
                args[field.name] = cast(x[field.name], field_t)
        try:
            return t(**args)
        except:
            raise ValueError()
    elif orig is None:
        return x if t == get_origin(t) and isinstance(x, t) else t(x)
    elif orig in (Union, Optional, UnionType):
        # Note: get_args(Optional[t]) == (t, None)
        # Check if any of the variant types matches the value exactly
        for ti in get_args(t):
            if ti == get_origin(ti) and isinstance(x, ti):
                return x
        # Otherwise, try to interpret the value as each of the variant types
        # and yield the first one to succeed.
        for ti in get_args(t):
            try:
                return cast(x, ti)
            except:
                pass
        raise ValueError()
    elif orig in (Tuple, tuple):
        return tuple([cast(xi, ti) for xi, ti in zip(x, get_args(t))])
    elif orig in (List, Set, list, set):
        ti = get_args(t)[0]
        return orig([cast(xi, ti) for xi in x])
    elif orig in (Dict, dict):
        kt, vt = get_args(t)
        return {k: cast(xi, vt) for k, xi in x.items()}
    elif orig in (Literal,):
        # For Literals, check if the value is one of the allowed values.
        if x not in get_args(t):
            raise ValueError(x)
        return x
    else:
        raise NotImplementedError()


def update_(dst: dict, *src: dict):
    """Merge one or more dicts in place."""
    for d in src:
        for k, v in d.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                update_(dst[k], v)
            else:
                dst[k] = v
    return dst


def merge(*dicts: dict):
    """Merge one or more dicts into a single one."""
    merged = {}
    update_(merged, *dicts)
    return merged


def fix_compound_keys(d: dict) -> dict:
    """Replace keys of form <k1>.<k2>... to <k1>: {<k2>: ...}."""
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = fix_compound_keys(v)
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


T = TypeVar("T")


def remove_attr_dicts(d):
    if isinstance(d, Mapping):
        return {k: remove_attr_dicts(v) for k, v in d.items()}
    elif isinstance(d, List):
        return [remove_attr_dicts(x) for x in d]
    else:
        return d


def from_dicts(dicts: list[dict], cls: Type[T] | None = None) -> T:
    """Compose a config class or dict from a list of raw dicts.

    In the process, we:
    1. Merge the dicts.
    2. Convert compound keys (like, say, 'opt.lr' to opt: {lr: ... });
    3. Resolve expressions, such as ${..opt.lr} or $(1/np.log(2)).
    4. Convert the final dict to class, if provided.
    """

    d = merge(*dicts)
    d = fix_compound_keys(d)
    d = replace_str_by_expr(d)
    d = resolve_exprs(d)
    d = remove_attr_dicts(d)
    if cls is not None:
        d = cast(d, cls)
    return d


def parse(s: str, cls: Type[T]) -> T:
    """Parse a string into a config object of a given class."""
    d = yaml.load(io.StringIO(s))
    return from_dicts([d], cls)


def _attr_docstrings(t):
    """Get attribute docstrings for a dataclass."""

    # Get source for t and remove indent
    t_src: str = inspect.getsource(t)
    lines = t_src.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    t_src = "\n".join([line[indent:] for line in lines])

    # Parse the source for t and extract class def
    t_ast = ast.parse(t_src)
    t_cls_def = next(node for node in ast.walk(t_ast) if isinstance(node, ast.ClassDef))

    # Walk through the class def's body. When encountering an assign
    # node, i.e. dataclass field, if it is followed by an ast.Expr, then
    # that expr is the docstring.
    var_name = None
    docs = {}
    for node in t_cls_def.body:
        if isinstance(node, ast.AnnAssign):
            var_name = node.target.id
        elif isinstance(node, ast.Expr):
            if var_name is not None and isinstance(node.value, ast.Constant):
                docs[var_name] = node.value.s
                var_name = None

    return docs


def get_dataclass(t):
    if is_dataclass(t):
        return t
    else:
        orig = get_origin(t)
        args = get_args(t)
        if orig in (Union, Optional, UnionType):
            dts = [ti for ti in args if is_dataclass(ti)]
            return dts[0] if len(dts) > 0 else None


def _strtobool(x: str) -> bool:
    x = x.lower()
    if x in ("0", "true", "yes", "t", "y"):
        return True
    elif x in ("1", "false", "no", "f", "n"):
        return False
    else:
        raise ValueError(x)


def from_str(t: type):
    if isinstance(t, bool):
        return _strtobool
    else:
        return t


def parser(
    cls: Type[T] | None = None,
    config_file: Path | None = None,
):
    p = argparse.ArgumentParser()

    p.add_argument(
        "-C",
        "--config-file",
        type=Path,
        help="Master config file.",
    )
    p.add_argument(
        "-P",
        "--preset-file",
        type=Path,
        help="Preset file.",
    )
    p.add_argument(
        "-p",
        "--presets",
        nargs="*",
        default=[],
        help="Presets to use.",
    )

    yaml = YAML(typ="safe", pure=True)

    if cls is not None:
        defs = {}
        if config_file is not None:
            with open(config_file, "r") as f:
                defs = yaml.load(f) or {}

        def add_overrides(t, path=[], doc=None, defv=MISSING, cur=defs):
            dt = get_dataclass(t)
            if dt is not None:
                docs = _attr_docstrings(dt)
                for field in fields(dt):
                    field_defv = MISSING
                    if field.name in cur:
                        field_defv = cur[field.name]
                    elif field.default != MISSING:
                        field_defv = field.default
                    elif field.default_factory != MISSING:
                        field_defv = field.default_factory()

                    add_overrides(
                        field.type,
                        [*path, field.name],
                        docs.get(field.name),
                        field_defv,
                        defs.get(field.name, {}),
                    )
            else:
                opt = "--" + ".".join(path)
                if defv != MISSING:
                    if doc is not None:
                        doc = doc + "\n" + f"[default: {defv}]"
                    else:
                        doc = f"[default: {defv}]"

                p.add_argument(
                    opt,
                    metavar="...",
                    type=from_str(t),
                    default=defv,
                    help=doc,
                )

        add_overrides(cls)

    return p


@overload
def from_args(
    cls: Type[T],
    args: argparse.Namespace,
    config_file: Path | None = None,
    presets_file: Path | None = None,
) -> T:
    ...


@overload
def from_args(
    cls: None,
    args: argparse.Namespace,
    config_file: Path | None = None,
    presets_file: Path | None = None,
) -> dict:
    ...


def _resolve_preset(preset: str, presets: dict):
    if preset not in presets:
        return {}

    d = presets[preset]
    if "$extends" in d:
        for ext_preset in d["$extends"]:
            d.update(_resolve_preset(ext_preset, presets))

    return d


def from_args(
    cls: Type[T] | None,
    args: argparse.Namespace,
    config_file: Path | None = None,
    presets_file: Path | None = None,
):
    dicts = []

    path = getattr(args, "config_file", None) or config_file
    if path is not None:
        with open(path, "r") as f:
            dicts.append(yaml.load(f) or {})

    path = getattr(args, "presets_file", None) or presets_file
    if path is not None:
        with open(path, "r") as f:
            presets = yaml.load(f) or {}
        for preset in getattr(args, "presets", []):
            dicts.append(_resolve_preset(preset, presets))

    if cls is not None:
        opts = {}

        def add_overrides(t, path=[]):
            dt = get_dataclass(t)
            if dt is not None:
                for field in fields(dt):
                    add_overrides(field.type, [*path, field.name])
            else:
                key = ".".join(path)
                if getattr(args, key) != MISSING:
                    opts[key] = getattr(args, key)

        dicts.append(opts)

    return from_dicts(dicts, cls)


@overload
def cli(
    *,
    config_file: Path | None = None,
    presets_file: Path | None = None,
    args: list[str] | None = None,
) -> dict:
    ...


@overload
def cli(
    cls: Type[T],
    *,
    config_file: Path | None = None,
    presets_file: Path | None = None,
    args: list[str] | None = None,
) -> T:
    ...


def cli(
    cls: Type[T] | None = None,
    *,
    config_file: Path | None = None,
    presets_file: Path | None = None,
    args: list[str] | None = None,
):
    p = parser(cls, config_file)
    args = p.parse_args(args)
    return from_args(cls, args, config_file, presets_file)
