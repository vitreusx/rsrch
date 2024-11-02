from typing import Any

from ..data import Buffer
from ..gym import VecAgent, VecEnv


class SDK:
    """Env SDK.

    Provides utilities for working with RL environments (doing rollouts, storing samples, fetching sequences etc.) without having to worry about converting the data to/from tensors - the SDK ensures that the data format given to the agent on rollout (see `rollout`), and retrieved from a wrapped buffer (see `wrap_buffer`) are identical.

    Another special feature is uniform presence of stat keys for all SDKs:
    - `total_steps`: Global (cross-episode) true (frame-skip-aware) step counter.
    - `ep_length`: Episode length, present on final episode steps. In Atari, if using episodic-life wrapper, indicates "true" episode length.
    - `ep_returns`: Episode returns. Behaves similarly to `ep_length`."""

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
