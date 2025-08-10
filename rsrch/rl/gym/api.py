from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Sequence, TypedDict, TypeVar

import numpy as np


class ObsType(TypedDict):
    obs: Any
    act: Any | None
    reward: float | None
    term: bool | None
    trunc: bool | None


T_obs = TypeVar("T_obs", bound=ObsType)
T_act = TypeVar("T_act")


class Agent(ABC, Generic[T_obs, T_act]):
    """An env agent API. Receives observations from the env, and returns an
    action given current knowledge of the environment state."""

    def __init__(self, obs_space: Any, act_space: Any):
        self.obs_space = obs_space
        self.act_space = act_space

    def reset(self, obs: T_obs):
        """Receive the first observation after an episode reset."""

    @abstractmethod
    def policy(self) -> T_act:
        """Get agent's action at a current time step."""

    def step(self, act: T_act, next_obs: T_obs):
        """Update agent's knowledge upon new time step, by passing performed
        action and the observation of the new state. NOTE: The action does not
        need to be the same, as the one computed by invoking `policy` method."""


class AgentWrapper(Agent[T_obs, T_act]):
    def __init__(self, agent: Agent):
        super().__init__(agent.obs_space, agent.act_space)
        self.agent = agent
        self.obs_space = agent.obs_space
        self.act_space = agent.act_space

    def reset(self, obs: T_obs):
        self.agent.reset(obs)

    def policy(self) -> T_act:
        return self.agent.policy()

    def step(self, act: T_obs, next_obs: T_act):
        self.agent.step(act, next_obs)


class VecAgent(ABC, Generic[T_obs, T_act]):
    """A vectorized version of `Agent`."""

    def __init__(self, obs_space: Any, act_space: Any):
        self.obs_space = obs_space
        self.act_space = act_space

    def reset(
        self,
        idxes: np.ndarray,
        obs_seq: Sequence[T_obs],
    ):
        """Notify agent of episode starts for the specified environments."""

    @abstractmethod
    def policy(self, idxes: np.ndarray) -> Sequence[T_act]:
        """Get agent's actions for specified environments."""

    def step(
        self,
        idxes: np.ndarray,
        act_seq: Sequence[T_act],
        next_obs_seq: Sequence[T_obs],
    ):
        """Update agent's knowledge states for specified environments."""


class VecAgentWrapper(VecAgent[T_obs, T_act]):
    def __init__(self, agent: VecAgent):
        super().__init__(agent.obs_space, agent.act_space)
        self.agent = agent

    def reset(self, idxes: np.ndarray, obs_seq):
        self.agent.reset(idxes, obs_seq)

    def policy(self, idxes: np.ndarray):
        return self.agent.policy(idxes)

    def step(self, idxes: np.ndarray, act_seq, next_obs_seq):
        self.agent.step(idxes, act_seq, next_obs_seq)


class Env(ABC, Generic[T_obs, T_act]):
    """An environment API. Represents a partially observable Markov decision
    process (POMDP)."""

    def __init__(self, obs_space: Any, act_space: Any):
        self.obs_space = obs_space
        self.act_space = act_space

    @abstractmethod
    def reset(self) -> T_obs:
        """Start a new episode.

        :return: A dict containing following fields:

        - `obs`: the initial observation;
        - possibly other data, depending on the env in question.
        """

    @abstractmethod
    def step(self, act: T_act) -> tuple[T_obs, bool]:
        """Perform an environment step with a given action.

        :return: A tuple `(step, final)`, where:

        - `step`: a dict containing following fields:

            - `act`: the action leading up to the new state;
            - `obs`: the observation of the new state;
            - `reward`: a reward obtained upon arriving at the state;
            - `term`, `trunc`: MDP termination and truncation signals.
            - possibly other data, depending on the env in question.

        - `final`: a boolean variable, denoting whether the new state is final
        (and thus whether `reset` call is necessary.
        """

    def rollout(self, agent: Agent[T_obs, T_act]) -> Iterable[tuple[T_obs, T_act]]:
        """Perform an environment rollout.

        Essentially, given an agent providing the actions to perform,
        automatically: (1) reset the env and the agent, (2) get actions from the
        agent, (3) perform env step, (4) update agent state on each step.

        :return: A stream of `(step, final)` tuples, where:

        - `step` is a dict with:

            - `obs` (observation of the current state);
            - `act`: the action leading up to the new state;
            - `reward`: a reward obtained upon arriving at the state;
            - `term`, `trunc`: MDP termination and truncation signals.
            - possibly other data, depending on the env in question.

        - `final` is a boolean variable, denoting whether the current
        state is final.
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


class EnvWrapper(Env[T_obs, T_act]):
    def __init__(self, env: Env):
        super().__init__(env.obs_space, env.act_space)
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, act):
        return self.env.step(act)


class VecEnv(Generic[T_obs, T_act], ABC):
    """A vectorized env.

    NOTE: Unlike simply using a set of `Env` instances, `VecEnv` implementation
    can be optimized, e.g. to perform in parallel. Because of this, `reset`
    and `step` methods are not available - one can use the environment by
    iterating over a `rollout` sequence.
    """

    def __init__(self, num_envs: int, obs_space: Any, act_space: Any):
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.act_space = act_space

    @abstractmethod
    def rollout(
        self, agent: VecAgent[T_obs, T_act]
    ) -> Iterable[tuple[int, tuple[T_obs, bool]]]:
        """Perform a rollout of a vectorized env, with a specified vectorized agent.

        :return: A stream of `(env_idx, (step, final))`, where `env_idx` is
        the environment index, and `(step, final)` is the same as in `Env.rollout`.
        """


class VecEnvWrapper(VecEnv[T_obs, T_act]):
    def __init__(self, env: VecEnv):
        super().__init__(env.num_envs, env.obs_space, env.act_space)
        self.env = env

    def rollout(self, agent):
        return self.env.rollout(agent)
