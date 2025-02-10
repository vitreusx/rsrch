from typing import Any

import torch
from torch import Tensor, nn


def all_submodules(module: nn.Module, path: str = ""):
    for name, child in module.named_children():
        child_path = name if path == "" else f"{path}.{name}"
        yield child_path, child
        yield from all_submodules(child, path=child_path)


class ExtractFeaturesHook:
    def __init__(self, module: nn.Module):
        self._module = module
        self.features: dict[str, Tensor] = {}
        self._hooks = []

        def submodule_hook(path: str):
            def hook(module: nn.Module, input: Any, output: Any):
                self.features[path] = output

            return hook

        for path, submodule in all_submodules(module):
            hook = submodule.register_forward_hook(submodule_hook(path))
            self._hooks.append(hook)

        def module_pre_hook(module: nn.Module, input: Any):
            self.features.clear()

        hook = module.register_forward_pre_hook(module_pre_hook)
        self._hooks.append(hook)

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.features.clear()

    def __del__(self):
        self.remove()


def extract_features_with_hook(module: nn.Module):
    """Extract intermediate features (i.e. outputs of module's submodules) from the forward calls of `module`.

    :param module: Torch module to be "hacked" to extract features.
    :returns: A pair `(features, hook)`, where:
    - `features` is a dictionary populated with the features;
    - `hook` has a method `remove`, which can be used to stop retrieving the features.
    """

    hook = ExtractFeaturesHook(module)
    return hook.features, hook
