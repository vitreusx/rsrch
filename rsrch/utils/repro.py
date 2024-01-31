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
    def set(self, seed: int, benchmark=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if benchmark:
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
