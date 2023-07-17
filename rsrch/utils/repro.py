import os
import random

import numpy as np
import torch


def set_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def use_deterministic(mode=True):
    torch.backends.cudnn.benchmark = mode
    torch.use_deterministic_algorithms(mode=mode)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_worker(worker_id):
    worker_seed: int = torch.initial_seed() % 2**32
    set_seeds(worker_seed)
