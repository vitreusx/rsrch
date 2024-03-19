from dataclasses import dataclass, field
from typing import Literal

from . import node as node_


@dataclass
class ExecEnv:
    type: Literal["node"] = "node"
    node: node_.Config = field(default_factory=node_.Config)


@dataclass
class Requires:
    exec_env: ExecEnv = field(default_factory=ExecEnv)
    remote: bool | None = None
    git_commit_sha: str | None = None
