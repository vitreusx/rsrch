import rsrch.distributions as D


class WorldModel:
    def next_pred(self, state, act) -> D.Distribution:
        """Get a distribution of s' given (s, a)."""

    def term_pred(self, state) -> D.Distribution:
        """Get a distribution of s being terminal."""

    def rew_pred(self, state) -> D.Distribution:
        """Get a distribution of r from s'. NOTE: We subsume r into s', which
        technically doesn't follow the definition of MDP, but it allows one to
        detach env dynamics from reward structure."""
