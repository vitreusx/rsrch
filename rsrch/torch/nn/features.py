import re
from typing import Any, Callable

from torch import Tensor, nn


class ExtractFeaturesHook:
    def __init__(
        self,
        module: nn.Module,
        pattern: Callable[[str, nn.Module], bool] | str | None = None,
    ):
        self._module = module
        self.features: dict[str, Tensor] = {}
        self._hooks = []

        def submodule_hook(path: str):
            def hook(module: nn.Module, input: Any, output: Any):
                self.features[path] = output

            return hook

        if pattern is None:
            pattern = lambda path, module: True
        elif isinstance(pattern, str):
            pattern_re = re.compile(f"^{pattern}$")
            pattern = lambda path, module: re.match(pattern_re, path) is not None

        for path, submodule in module.named_modules():
            if pattern is None or pattern(path, submodule):
                hook = submodule.register_forward_hook(submodule_hook(path))
                self._hooks.append(hook)

        def module_pre_hook(module: nn.Module, input: Any):
            self.features.clear()

        hook = module.register_forward_pre_hook(module_pre_hook)
        self._hooks.append(hook)

    def remove(self):
        """Remove/detach the hook."""

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.features.clear()

    def __del__(self):
        self.remove()


def extract_features_with_hook(
    module: nn.Module,
    pattern: Callable[[str, nn.Module], bool] | str | None = None,
):
    """Extract intermediate features (i.e. outputs of module's submodules)
    from the forward calls of `module`.

    :param module: Torch module to be "hacked" to extract features.
    :param pattern: Optional pattern for matching submodules.

        - If it's a string, it's assumed to be a regex pattern.
        - If it's callable, it's used as a filter function
        (`pattern(path, module) -> bool`).
        - If not provided or `None`, all submodules are matched.

    :return: A pair `(features, hook)`, where:

        - `features` is a dictionary populated with the features;
        - `hook` object has a method `remove`, which can be used to stop
        retrieving the features.
    """

    hook = ExtractFeaturesHook(module, pattern)
    return hook.features, hook


class ExtractGradientsHook:
    def __init__(
        self,
        module: nn.Module,
        pattern: Callable[[str], bool] | str | None = None,
    ):
        self._module = module
        self.gradients: dict[str, Tensor] = {}
        self._hooks = []

        def param_hook(path: str):
            def hook(grad: Tensor):
                self.gradients[path] = grad

            return hook

        if pattern is None:
            pattern = lambda path: True
        elif isinstance(pattern, str):
            pattern_re = re.compile(f"^{pattern}$")
            pattern = lambda path: re.match(pattern_re, path) is not None

        for path, param in module.named_parameters():
            if pattern is None or pattern(path):
                hook = param.register_hook(param_hook(path))
                self._hooks.append(hook)

        def module_pre_hook(module: nn.Module, grad_output: Any):
            self.gradients.clear()

        hook = module.register_full_backward_pre_hook(module_pre_hook)
        self._hooks.append(hook)

    def remove(self):
        """Remove/detach the hook."""

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.gradients.clear()

    def __del__(self):
        self.remove()


def extract_gradients_with_hook(
    module: nn.Module,
    pattern: Callable[[str], bool] | str | None = None,
):
    hook = ExtractGradientsHook(module, pattern)
    return hook.gradients, hook
