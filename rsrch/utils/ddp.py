import math
import os
from typing import Any, Callable, Iterable, Literal, Sequence, Sized, TypeVar

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler, SequentialSampler

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


class DistributedSampler:
    """A more generic version of `torch.utils.data.DistributedSampler`. Makes any (sized) sampler, including batch samplers, a distributed sampler.

    Note: Torch's variant has a `shuffle` option, which is missing here. You need to provide a random sampler, if you want to replicate the behavior of `shuffle=True`.
    """

    def __init__(
        self,
        sampler: Sized,
        set_epoch: Callable[[int], None],
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.set_epoch = set_epoch
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        index_count = len(self.sampler)
        if self.drop_last:
            self.num_samples = index_count // self.num_replicas
        else:
            self.num_samples = (
                index_count + self.num_replicas - 1
            ) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas
        self.padding_size = self.total_size - index_count

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        pad = []

        local = 0
        for index in self.sampler:
            if local >= self.total_size:
                break
            if local < self.padding_size:
                pad.append(index)
            if local % self.num_replicas == self.rank:
                yield index
            local += 1

        for index in pad:
            if local >= self.total_size:
                break
            if local % self.num_replicas == self.rank:
                yield index
            local += 1


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
        self.num_replicas = torch.distributed.get_world_size()
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

    def wrap_sampler(
        self,
        sampler: Sized,
        set_epoch: Callable[[int], None],
        drop_last: bool = False,
    ):
        """Make an index or a batch sampler ready for use in DDP.

        Makes rank :math:`r` process only process samples :math:`r+kN` where :math:`N` is the world size.

        **Warning**: Remember to set DDP sampler epoch for shuffled samplers, to
        ensure consistency between DDP processes.
        """

        return DistributedSampler(
            sampler=sampler,
            set_epoch=set_epoch,
            num_replicas=self.num_replicas,
            rank=self.rank,
            drop_last=drop_last,
        )

    def set_epoch(self, sampler: DistributedSampler, epoch: int):
        """Sets current epoch number for a sampler obtained via `wrap_sampler`."""
        sampler.set_epoch(epoch)

    def all_reduce(self, tensor: Tensor, op: ReduceOpType):
        """Perform in-place all-reduce op on a tensor."""
        op = getattr(ReduceOp, op.upper())
        torch.distributed.all_reduce(tensor, op)


class SPFallback:
    """An API-compatible single-process fallback for `DDPHelper`."""

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

    def wrap_sampler(
        self,
        sampler: Sized,
        set_epoch: Callable[[int], None],
        drop_last: bool = False,
    ):
        return DistributedSampler(
            sampler=sampler,
            set_epoch=set_epoch,
            num_replicas=1,
            rank=0,
            drop_last=drop_last,
        )

    def set_epoch(self, sampler: DistributedSampler, epoch: int):
        sampler.set_epoch(epoch)

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
