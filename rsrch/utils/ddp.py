import os
from typing import Literal, Sequence, Sized, TypeVar

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

M = TypeVar("M")
F = TypeVar("F")

ReduceOpType = Literal[
    "sum",
    "avg",
    "product",
    "min",
    "max",
    "band",
    "bor",
    "bxor",
    "premul_sum",
]


class DDPHelper:
    """A helper for Pytorch's DDP."""

    def __init__(self):
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("DDPHelper requires LOCAL_RANK to be set.")

        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            backend = "nccl"
            self.device_index = local_rank % torch.cuda.device_count()
            device = f"cuda:{self.device_index}"
        else:
            backend = "gloo"
            device = "cpu"

        torch.distributed.init_process_group(backend=backend)
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device = torch.device(device)

    def wrap_model(self, model: M) -> M:
        """Prepare a model for use in DDP."""
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model)
        return model

    def state_dict(self, model: DistributedDataParallel) -> dict[str, Tensor]:
        """Get a state dict for a wrapped model."""
        unwrapped = model.module
        return unwrapped.state_dict()

    @property
    def is_master(self):
        """Whether the current process is the master process."""
        return self.rank == 0

    def get_sampler(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        """Prepare a sampler for use in `DataLoader`."""
        return DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def set_epoch(
        self,
        sampler: DistributedSampler,
        epoch: int,
    ):
        """Set epoch number for a sampler obtained using `get_sampler`."""
        sampler.set_epoch(epoch)

    def all_reduce(self, tensor: Tensor, op: ReduceOpType):
        """Perform in-place all-reduce op on a tensor."""
        op = getattr(ReduceOp, op.upper())
        torch.distributed.all_reduce(tensor, op)


class SPFallback:
    """An API-compatible single-process fallback for `ViaDDP`."""

    def __init__(self, device: str | torch.device | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def wrap_model(self, model: M) -> M:
        return model.to(self.device)

    def state_dict(self, model: nn.Module):
        return model.state_dict()

    @property
    def is_master(self):
        return True

    def get_sampler(
        self,
        dataset: Sized,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if shuffle:
            gen = torch.Generator().manual_seed(seed)
            return RandomSampler(dataset, replacement=False, generator=gen)
        else:
            return SequentialSampler(dataset)

    def set_epoch(
        self,
        sampler: DistributedSampler,
        epoch: int,
    ):
        pass

    def all_reduce(self, tensor: Tensor, op: ReduceOpType):
        pass


def auto_detect() -> DDPHelper:
    """Auto-detect and setup "infrastructure" (DDP etc.)

    Depending on the environment variables. If using multiple processes (as indicated by `LOCAL_RANK` env variable), DDP is used, and an appropriate helper is returned. Otherwise, API-compatible version for single-process setup is returned. This is done in order to streamline single- and multi-process deployments.
    """

    if "LOCAL_RANK" in os.environ:
        return DDPHelper()
    else:
        return SPFallback()
