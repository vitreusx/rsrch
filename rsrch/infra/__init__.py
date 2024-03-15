import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import git
import psutil
import torch


@dataclass
class Requires:
    distributed: bool = False
    num_procs: int = 1
    procs_per_node: int = 1
    ram_per_proc: float | None = None
    remote_only: bool = False
    gpus_per_proc: int = 0
    vram_per_gpu: float | None = None
    cuda_only: bool = True
    datasets: list[str] = field(default_factory=lambda: [])
    git_commit_sha: str | None = None


class LocalEnv:
    def __init__(self):
        self.repo = git.Repo(
            path=Path(__file__).parent,
            search_parent_directories=True,
        )

        # Fetch env vars for distributed launch via torchrun
        self.is_distributed = "WORLD_SIZE" in os.environ
        self.num_procs = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.procs_per_node = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    def satisfies(self, req: Requires) -> str | None:
        if req.remote_only:
            return "Cannot run locally."

        if req.distributed:
            if not self.is_distributed:
                return "Not running a distributed job."

            if req.num_procs != self.num_procs:
                return "Invalid # of procs."

            if req.procs_per_node != self.procs_per_node:
                return "Invalid # of local procs."

        if req.ram_per_proc is not None:
            ram_per_node = req.procs_per_node * req.ram_per_proc
            if psutil.virtual_memory().available < ram_per_node:
                return "Not enough RAM available."

        if req.gpus_per_proc > 0:
            assert req.cuda_only
            gpus_per_node = req.gpus_per_proc * req.procs_per_node
            if torch.cuda.device_count() < gpus_per_node:
                return "Not enough CUDA devices available."
            if req.vram_per_gpu is not None:
                for idx in range(gpus_per_node):
                    free, total = torch.cuda.mem_get_info(idx)
                    if free < req.vram_per_gpu:
                        return "Not enough vRAM available."

        if req.git_commit_sha is not None:
            if self.repo.is_dirty():
                return "Git repo is dirty."
            if self.repo.head.commit.hexsha != req.git_commit_sha:
                return "Invalid Git HEAD commit SHA."


class RemoteSSH:
    def __init__(self, host: str, path: str):
        self.host = host
        self.path = path

    def fork(self):
        ...
