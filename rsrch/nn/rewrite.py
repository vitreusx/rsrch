from typing import Protocol

from torch import nn


class RewriteFn(Protocol):
    def __call__(self, fqname: str, module: nn.Module) -> nn.Module:
        """Rewrite (non-recursively) a module.
        :param fqname: Fully-qualified module name.
        :param module: Module to rewrite.
        :return: Rewritten module."""


def rewrite_module_(
    module: nn.Module,
    func: RewriteFn,
    recursive=True,
    fqname: str | None = None,
):
    mod: nn.Module = func(fqname, module)
    if recursive:
        for name, child in module.named_children():
            child_fqname = name if fqname is None else f"{fqname}.{name}"
            mod.add_module(name, rewrite_module_(child, func, True, child_fqname))
    return mod
