import os
import random

import numpy as np

HAS_TORCH = True
try:
    import torch
except ModuleNotFoundError:
    HAS_TORCH = False


def seed_all(seed: int):
    """Seed global RNGs."""
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def set_fully_deterministic(mode: bool = True):
    """Set PyTorch to be fully deterministic."""
    torch.backends.cudnn.benchmark = not mode
    torch.use_deterministic_algorithms(mode)
    if mode:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def worker_init_fn(seed: int):
    """Create a `worker_init_fn` to be passed to dataloaders. Each worker gets
    a different seed."""

    def func(worker_id: int):
        worker_seed: int = (seed + worker_id) % 2**32
        seed_all(worker_seed)
        if HAS_TORCH:
            set_fully_deterministic(not torch.backends.cudnn.benchmark)

    return func


class RandomState:
    """Random state proxy for Python, Numpy and Pytorch."""

    @staticmethod
    def save():
        state = {
            "np": np.random.get_state(),  # noqa: NPY002
            "random": random.getstate(),
        }
        if HAS_TORCH:
            state = {
                **state,
                "torch_cpu": torch.get_rng_state().numpy(),
                "torch_cuda": [x.cpu().numpy() for x in torch.cuda.get_rng_state_all()],
            }
        return state

    @staticmethod
    def load(state: dict):
        np.random.set_state(state["np"])  # noqa: NPY002
        random.setstate(state["random"])
        if HAS_TORCH:
            torch.set_rng_state(torch.as_tensor(state["torch_cpu"]))
            for idx, rs in enumerate(state["torch_cuda"]):
                device = torch.device(f"cuda:{idx}")
                torch.cuda.set_rng_state(torch.as_tensor(rs), device)


state = RandomState()
