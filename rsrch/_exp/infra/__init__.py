from pathlib import Path

import git
import torch
from ruamel.yaml import YAML

from . import node
from .config import Requires

yaml = YAML(typ="safe", pure=True)


def current_repo():
    return git.Repo(
        path=Path(__file__).parent,
        search_parent_directories=True,
    )


def try_ensure(req: Requires):
    """Try to ensure that the provided requirements are satisfied."""

    if req.remote:
        raise RuntimeError("Cannot run locally.")

    if req.git_commit_sha is not None:
        repo = current_repo()

        if repo.is_dirty():
            raise RuntimeError("Git repo cannot be dirty.")

        if repo.head.commit.hexsha != req.git_commit_sha:
            # Here we can try to check out the commit
            raise RuntimeError("Switching Git HEAD not supported.")

    if req.exec_env.type == "node":
        env_cfg = req.exec_env.node

        if env_cfg.device not in ("cpu", "cuda"):
            raise RuntimeError("Accelerator type not supported.")

        if env_cfg.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available.")

        return node.ExecEnv(env_cfg)
    else:
        raise RuntimeError("Exec env type not supported.")
