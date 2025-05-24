import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SingleProcess:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)

    @property
    def is_master(self):
        return True

    def wrap_model(self, model: nn.Module):
        return model.to(self.device)

    def create_data_loader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        dist_seed: int = 0,
        drop_last: bool = False,
        **kwargs,
    ):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            **kwargs,
        )


class ViaDDP:
    def __init__(self, backend: str = "nccl"):
        torch.distributed.init_process_group(backend)

        self.rank = torch.distributed.get_rank()

        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")

    @property
    def is_master(self):
        return self.rank == 0

    def wrap_model(self, model: nn.Module):
        return DDP(model, [self.device_id])

    def create_data_loader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        dist_seed: int = 0,
        drop_last: bool = False,
        **kwargs,
    ):
        return DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                shuffle=shuffle,
                seed=dist_seed,
                drop_last=drop_last,
            ),
            drop_last=drop_last,
            **kwargs,
        )


def auto_detect():
    if "LOCAL_RANK" in os.environ:
        return ViaDDP()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SingleProcess(device)
