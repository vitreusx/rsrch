import rsrch.distributions as D
from rsrch.rl import gym


class WorldModel:
    def step(self, s, a) -> D.Distribution:
        """Get a distribution of next state s'."""
        ...

    def term(self, s) -> D.Distribution:
        """Get a distribution over terminality of state s."""
        ...

    def reward(self, next_s) -> D.Distribution:
        """Get a distribution over reward granted upon entering next state s'."""
        ...


class Actor:
    def __call__(self, s) -> D.Distribution:
        """Get a policy in current state s. The output actions are encoded."""
        ...


class Critic:
    def __call__(self, s) -> D.Distribution:
        """Get a distribution over state values for state s."""
        ...
