from dataclasses import dataclass
from typing import Any, Iterable, Literal, Protocol

from rsrch import spaces

from ..data import Buffer
from ..gym import VecAgent, VecEnv
from . import atari, dmc, gym, wrappers


class SDK:
    """Env SDK. Provides utilities for working with RL environments (doing rollouts, storing samples, fetching sequences etc.) without having to worry about converting the data to/from tensors - the SDK ensures that the data format given to the agent on rollout (see `rollout`), and retrieved from a wrapped buffer (see `wrap_buffer`) are identical."""

    id: str
    """Env id string."""

    obs_space: Any
    """Observation space, in target/tensor format."""

    act_space: Any
    """Action space, in target/tensor format."""

    def make_envs(self, num_envs: int, **kwargs) -> VecEnv:
        """Create a vector env."""

    def wrap_buffer(self, buffer: Buffer) -> Buffer:
        """Wrap a regular buffer into an SDK-aware version. Episodes in the buffer are automatically converted to target format on retrieval."""

    def rollout(self, envs: VecEnv, agent: VecAgent):
        """Create a rollout with vector env `envs` and vector agent `agent`. The agent must operate in target format; the actions and observations in the env format are converted automatically."""


AtariCfg = atari.Config
GymCfg = gym.Config
DMCCfg = dmc.Config


@dataclass
class Config:
    type: Literal["atari", "gym", "dmc"]
    atari: AtariCfg | None = None
    gym: GymCfg | None = None
    dmc: DMCCfg | None = None


def make(cfg: Config) -> SDK:
    """Create an SDK from a config object."""
    if cfg.type == "atari":
        return atari.SDK(cfg.atari)
    elif cfg.type == "gym":
        return gym.SDK(cfg.gym)
    elif cfg.type == "dmc":
        return dmc.SDK(cfg.dmc)
