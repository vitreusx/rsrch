from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import numpy as np


class Agent(ABC):
    """An env agent API. Receives observations from the env, and returns an action given current knowledge of the environment state."""

    obs_space: Any
    act_space: Any

    def reset(self, obs: Any):
        """Receive the first observation after an episode reset."""

    @abstractmethod
    def policy(self):
        """Get agent's action at a current time step."""

    def step(self, act: Any, next_obs: Any):
        """Update agent's knowledge upon new time step, by passing performed action and the observation of the new state. NOTE: The action does not need to be the same, as the one computed by invoking `policy` method."""


class AgentWrapper(Agent):
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
        self.obs_space = agent.obs_space
        self.act_space = agent.act_space

    def reset(self, step):
        self.agent.reset(step)

    def policy(self):
        return self.agent.policy()

    def step(self, act, next_obs):
        self.agent.step(act, next_obs)


class VecAgent(ABC):
    """A vectorized version of `Agent`."""

    obs_space: Any
    act_space: Any

    def reset(self, idxes: np.ndarray, obs_seq: Sequence[Any]):
        """Notify agent of episode starts for the specified environments."""

    @abstractmethod
    def policy(self, idxes: np.ndarray):
        """Get agent's actions for specified environments."""

    def step(
        self,
        idxes: np.ndarray,
        act_seq: Sequence[Any],
        next_obs_seq: Sequence[Any],
    ):
        """Update agent's knowledge states for specified environments."""


class VecAgentWrapper(VecAgent):
    def __init__(self, agent: VecAgent):
        super().__init__()
        self.agent = agent
        self.obs_space = agent.obs_space
        self.act_space = agent.act_space

    def reset(self, idxes: np.ndarray, obs_seq):
        self.agent.reset(idxes, obs_seq)

    def policy(self, idxes: np.ndarray):
        return self.agent.policy(idxes)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self.agent.step(idxes, act_seq, next_obs_seq)


class Env(ABC):
    """An environment API. Represents a partially observable Markov decision process (POMDP)."""

    obs_space: Any
    act_space: Any

    @abstractmethod
    def reset(self) -> dict:
        """Start a new episode.

        :return: A dict containing following fields:

        - `obs`: the initial observation;
        - possibly other data, depending on the env in question.
        """

    @abstractmethod
    def step(self, act) -> tuple[dict, bool]:
        """Perform an environment step with a given action.

        :return: A tuple `(step, final)`, where:

        - `step`: a dict containing following fields:
            - `act`: the action leading up to the new state;
            - `obs`: the observation of the new state;
            - `reward`: a reward obtained upon arriving at the state;
            - `term`, `trunc`: MDP termination and truncation signals.
            - possibly other data, depending on the env in question.
        - `final`: a boolean variable, denoting whether the new state is final (and thus whether `reset` call is necessary.
        """

    def rollout(self, agent: Agent) -> Iterable[tuple[dict, bool]]:
        """Perform an environment rollout.

        Essentially, given an agent providing the actions to perform, automatically: (1) reset the env and the agent, (2) get actions from the agent, (3) perform env step, (4) update agent state on each step.

        :return: A stream of `(step, final)` pairs, where:

        - `step`: a dict containing following fields:
            - `obs`: an observation of the current state;
            - if step is non-initial:

                - `act`: the action leading up to the new state;
                - `reward`: a reward obtained upon arriving at the state;
                - `term`, `trunc`: MDP termination and truncation signals.
            - possibly other data, depending on the env in question.
        - `final` is a boolean variable, denoting whether the current state is final.
        """

        obs = None
        while True:
            if obs is None:
                obs = self.reset()
                agent.reset(obs)
                yield obs, False

            act = agent.policy()
            next_obs, final = self.step(act)
            agent.step(act, next_obs)
            yield {**next_obs, "act": act}, final

            obs = next_obs
            if final:
                obs = None


class EnvWrapper(Env):
    def __init__(self, env: Env):
        super().__init__()
        self.env = env
        self.obs_space = env.obs_space
        self.act_space = env.act_space

    def reset(self):
        return self.env.reset()

    def step(self, act):
        return self.env.step(act)


class VecEnv(ABC):
    """A vectorized env.

    Unlike simply using a set of `Env` instances, `VecEnv` implementation can be optimized, e.g. to perform in parallel. Because of this, `reset` and `step` methods are not available - one can use the environment by iterating over a `rollout` sequence.
    """

    num_envs: int
    obs_space: Any
    act_space: Any

    @abstractmethod
    def rollout(self, agent: VecAgent) -> Iterable[tuple[int, tuple[dict, bool]]]:
        """Perform a rollout of a vectorized env, with a specified vectorized agent.

        :return: A stream of `(env_idx, (step, final))`, where `env_idx` is the environment index, and `(step, final)` is the same as in `Env.rollout`.
        """


class VecEnvWrapper(VecEnv):
    def __init__(self, env: VecEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.act_space = env.act_space
        self.obs_space = env.obs_space

    def rollout(self, agent):
        return self.env.rollout(agent)
