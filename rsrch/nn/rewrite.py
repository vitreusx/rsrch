from torch import nn


def rewrite_module(module: nn.Module, func, recursive=True, fqdn=None):
    mod = func(fqdn, module)
    if recursive:
        for name, child in module.named_children():
            fqdn_ = name if fqdn is None else f"{fqdn}.{name}"
            mod.add_module(name, rewrite_module(child, func, True, fqdn_))
    return mod
