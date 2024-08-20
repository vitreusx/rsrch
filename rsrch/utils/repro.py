import os
import random

import numpy as np
import torch


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_fully_deterministic(mode=True):
    torch.backends.cudnn.benchmark = not mode
    torch.use_deterministic_algorithms(mode)
    if mode:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def worker_init_fn(worker_id):
    worker_seed: int = (torch.initial_seed() + worker_id) % 2**32
    seed_all(worker_seed)
    set_fully_deterministic(not torch.backends.cudnn.benchmark)


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
