class Specs:
    """Environment specs."""


class Infra:
    @property
    def exec_env(self):
        """Current execution environment."""

    def ensure(self, specs: Specs):
        """Ensure that the current environment conforms to given specs. If not, then a fork of the current program is created in one of the available environments."""
