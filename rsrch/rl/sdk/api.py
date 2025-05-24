from typing import Any

from ..data import Buffer
from ..gym import VecAgent, VecEnv


class SDK:
    """RL environment SDK.

    Provides utilities for working with RL environments (doing rollouts, storing samples, fetching sequences etc.) without having to worry about converting the data to/from tensors - the SDK ensures that the data format given to the agent on rollout (see `rollout`), and retrieved from a wrapped buffer (see `wrap_buffer`) are identical.

    These functionalities allow one to greatly simplify env-agent interaction loop. We provide an example one below:

    .. code-block:: python
        # Initialize envs, vec agent and buffer to store data.
        envs = sdk.make_envs(num_envs, ...)
        agent = ...
        buf = sdk.wrap_buffer(data.Buffer())

        # Maintain a sequence ID for each environment.
        seq_ids = default_dict(lambda: None)

        # Perform rollout
        for env_idx, (step, final) in sdk.rollout(envs, agent):
            seq_ids[env_idx] = buf.push(seq_ids[env_idx], step, final)

    where `agent` operates on tensors already. Details such as resets and updates of both environments and agents, or the parallel implementation of vectorized environments, are all hidden from the user.
    """

    id: str
    """Environment identifier."""

    obs_space: Any
    """Observation space, in target format."""

    act_space: Any
    """Action space, in target format."""

    def make_envs(self, num_envs: int, **kwargs) -> VecEnv:
        """Create a vector env.

        :param num_envs: Number of parallel environments.
        :param kwargs: Parameters for the environments. The specific list of parameters depends on the SDK in question.
        """

    def wrap_buffer(self, buffer: Buffer) -> Buffer:
        """Wrap a regular buffer into an SDK-aware version. Episodes in the buffer are automatically converted to target format (e.g. `torch` tensors) on retrieval."""

    def rollout(self, envs: VecEnv, agent: VecAgent):
        """Create a rollout with vector env `envs` and vector agent `agent`. The agent must operate in target format; the actions and observations in the env format are converted automatically."""
