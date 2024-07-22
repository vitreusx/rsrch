from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from rsrch import spaces

from .. import buffer, env
from . import atari


@dataclass
class Config:
    type: Literal["atari"]
    atari: atari.Config


class Helper(Protocol):
    obs_space: spaces.torch.Space
    act_space: spaces.torch.Space

    def make_envs(self, num_envs: int, **kwargs) -> list[env.Env]:
        ...

    def wrap_buffer(self, buf: buffer.Buffer) -> buffer.Buffer:
        ...

    def rollout(
        self, envs: list[env.Env], agent: env.VecAgent
    ) -> Iterable[tuple[int, tuple[dict, bool]]]:
        ...


def make(cfg: Config) -> Helper:
    if cfg.type == "atari":
        return atari.Helper(cfg.atari)
