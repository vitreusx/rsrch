import os
import random

import numpy as np
import torch


def fix_seeds(seed: int, deterministic=False):
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(worker_id):
    worker_seed: int = (torch.initial_seed() + worker_id) % 2**32
    deterministic = not torch.backends.cudnn.benchmark
    fix_seeds(worker_seed, deterministic)


class RandomState:
    """Random state manager for Python, Numpy and Pytorch."""

    @staticmethod
    def save():
        return {
            "np": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().numpy(),
            "torch_cuda": [x.cpu().numpy() for x in torch.cuda.get_rng_state_all()],
            "random": random.getstate(),
        }

    @staticmethod
    def load(state: dict):
        np.random.set_state(state["np"])
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"]))
        for idx, rs in enumerate(state["torch_cuda"]):
            device = torch.device(f"cuda:{idx}")
            torch.cuda.set_rng_state(torch.as_tensor(rs), device)
        random.setstate(state["random"])


state = RandomState()
