from collections import namedtuple
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from ruamel.yaml import YAML
from rsrch.utils.config import from_dicts
import numpy as np
import git
import psutil
import torch
import paramiko

yaml = YAML(typ="safe", pure=True)


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
    python_env = None

    def __post_init__(self):
        assert self.num_procs % self.procs_per_node == 0
        self.distributed |= self.num_procs > 1


def get_repo():
    return git.Repo(
        path=Path(__file__).parent,
        search_parent_directories=True,
    )


class CurrentEnv:
    def __init__(self):
        self.repo = get_repo()
        self._req: Requires = None

        # Fetch env vars for distributed launch via torchrun
        self.is_distributed = "WORLD_SIZE" in os.environ
        self.num_procs = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.procs_per_node = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    @property
    def devices(self):
        if self._req.gpus_per_proc > 0:
            idxes = np.arange(torch.cuda.device_count())
            idxes = idxes.reshape(-1, self._req.gpus_per_proc)
            return [torch.device("cuda", i) for i in idxes[self.local_rank]]
        else:
            return [torch.device("cpu")]

    @property
    def master(self):
        return self.rank == 0

    def ensure(self, req: Requires):
        """Ensure that the environment satisfies a list of requirements."""

        if req.remote_only:
            raise RuntimeError("Cannot run locally.")

        if req.distributed:
            if not self.is_distributed:
                raise RuntimeError("Not running a distributed job.")

            if req.num_procs != self.num_procs:
                raise RuntimeError("Invalid # of procs.")

            if req.procs_per_node != self.procs_per_node:
                raise RuntimeError("Invalid # of local procs.")

        if req.ram_per_proc is not None:
            ram_per_node = req.procs_per_node * req.ram_per_proc
            if psutil.virtual_memory().available < ram_per_node:
                raise RuntimeError("Not enough RAM available.")

        if req.gpus_per_proc > 0:
            if not req.cuda_only:
                raise RuntimeError("Only CUDA is currently supported.")

            gpus_per_node = req.gpus_per_proc * req.procs_per_node
            if torch.cuda.device_count() < gpus_per_node:
                raise RuntimeError("Not enough CUDA devices available.")
            if req.vram_per_gpu is not None:
                for idx in range(gpus_per_node):
                    free, total = torch.cuda.mem_get_info(idx)
                    if free < req.vram_per_gpu:
                        raise RuntimeError("Not enough vRAM available.")

        if req.git_commit_sha is not None:
            if self.repo.is_dirty():
                raise RuntimeError("Git repo is dirty.")

            if self.repo.head.commit.hexsha != req.git_commit_sha:
                # Here we can try to check out the commit
                raise NotImplementedError()

        self._req = req


class RemoteSSH:
    def __init__(self, path: str, ssh_params: dict = {}):
        self.path = path
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(**ssh_params)

    def fork(self):
        ...


@dataclass
class Ensure:
    env: CurrentEnv
    is_local_env: bool
    is_exec_env: bool
    remote_info: dict | None = None


def ensure(req: Requires | None, env: dict) -> Ensure:
    """Ensure that requirements are satisfied. If they are not satisfied,
    fork the experiment to a host or deployment which can support it."""

    if req is None:
        req = Requires()

    cur_env = CurrentEnv()
    try:
        cur_env.ensure(req)
        return Ensure(cur_env, True, True)
    except:
        raise NotImplementedError()
