import os
import random

import numpy as np
import torch


def fix_seeds(seed: int, benchmark=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if benchmark:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_worker(worker_id):
    worker_seed: int = (torch.initial_seed() + worker_id) % 2**32
    fix_seeds(worker_seed, benchmark=torch.backends.cudnn.benchmark)


class RandomState:
    """Random state manager for Python, Numpy and Pytorch."""

    def init(self, seed: int, deterministic=False):
        torch.backends.cudnn.benchmark = not deterministic
        torch.use_deterministic_algorithms(deterministic)
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def save(self):
        return {
            "np": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().numpy(),
            "torch_cuda": [x.cpu().numpy() for x in torch.cuda.get_rng_state_all()],
            "random": random.getstate(),
        }

    def restore(self, state):
        np.random.set_state(state["np"])
        torch.set_rng_state(torch.as_tensor(state["torch_cpu"]))
        for idx, rs in enumerate(state["torch_cuda"]):
            device = torch.device(f"cuda:{idx}")
            torch.cuda.set_rng_state(torch.as_tensor(rs, device=device), device)
        random.setstate(state["random"])
