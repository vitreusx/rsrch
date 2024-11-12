import io
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from rsrch.nn.utils import safe_mode


def save_ref_state(module: nn.Module) -> dict[str, Tensor]:
    state = module.state_dict()
    return {name: tensor.data.clone() for name, tensor in state.items()}


@torch.no_grad()
def weight_norm_test(cur_state: dict, ref_state: dict):
    """Compute weight magnitudes of the parameters."""

    results = {}
    for name in cur_state:
        if not isinstance(cur_state[name], Tensor):
            continue

        ref_p: Tensor = ref_state[name]

        p_res = {}

        cur_p: Tensor = cur_state[name]
        cur_norm = torch.linalg.norm(cur_p).item()
        p_res["cur_norm"] = cur_norm

        diff = cur_p - ref_p
        diff_norm = torch.linalg.norm(diff).item()
        p_res["diff_norm"] = diff_norm

        ref_norm = torch.linalg.norm(ref_p).item()
        if ref_norm > 0:
            p_res["rel_norm"] = cur_norm / ref_norm
            p_res["rel_diff_norm"] = diff_norm / ref_norm

        results[name] = p_res

    return results


def weight_norm_metrics(results: dict):
    metrics = {}
    for key in ("cur_norm", "diff_norm", "rel_norm", "rel_diff_norm"):
        values = [r[key] for r in results.values() if key in r]
        if len(values) > 0:
            metrics[key] = np.mean(values)
    return metrics


def subpath(parent: str, name: str):
    return name if parent == "" else f"{parent}.{name}"


def named_apply_(module: nn.Module, func, path: str = ""):
    func(module, path)
    for name, child in module.named_children():
        named_apply_(child, func, subpath(path, name))


def dead_units_test(
    module: nn.Module,
    input,
    elu_eps: float = 1e-2,
):
    """Compute the number of dead units, defined as the number of activation units which output only zeros (for ReLU) or extreme values."""

    handles = []
    results = {}

    def func(module: nn.Module, path: str):
        if isinstance(module, nn.ReLU):

            def hook(relu, input, output: Tensor):
                is_dead = (output == 0).all(0)
                results[path] = {
                    "count": is_dead.count_nonzero(),
                    "total": is_dead.numel(),
                }

        elif isinstance(module, nn.ELU):

            def hook(elu: nn.ELU, input, output: Tensor):
                is_dead = (output < elu.alpha * (elu_eps - 1.0)).all(0)
                results[path] = {
                    "count": is_dead.count_nonzero().item(),
                    "total": is_dead.numel(),
                }

            handles.append(module.register_forward_hook(hook))

    named_apply_(module, func)

    with safe_mode(module):
        if isinstance(input, tuple):
            if isinstance(input[-1], dict):
                args, kwargs = input[:-1], input[-1]
            else:
                args, kwargs = input, {}
        else:
            args, kwargs = (input,), {}
        output = module(*args, **kwargs)

    for handle in handles:
        handle.remove()

    return (output, results)


def dead_unit_metrics(results: dict):
    metrics = {}
    values = [r["count"] / r["total"] for r in results.values()]
    if len(values) > 0:
        metrics["avg_freq"] = np.mean(values)
    return metrics


def full_test(module: nn.Module, ref_state: dict, input):
    cur_state = module.state_dict()
    wt_res = weight_norm_test(cur_state, ref_state)
    output, dead_res = dead_units_test(module, input)
    results = {"weight_norm": wt_res, "dead_units": dead_res}
    return output, results


def full_metrics(results):
    wt_mets = weight_norm_metrics(results["weight_norm"])
    dead_mets = dead_unit_metrics(results["dead_units"])
    return {
        **{f"weight_norm/{k}": v for k, v in wt_mets.items()},
        **{f"dead_units/{k}": v for k, v in dead_mets.items()},
    }
