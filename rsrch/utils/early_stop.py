class EarlyStopping:
    """Early stopping monitor.

    NOTE: The loss value must be always positive for it to work properly."""

    def __init__(
        self,
        rel_patience: float = 0.5,
        margin: float = 0.0,
        min_steps: int = 0,
        max_steps: int | None = None,
    ):
        """Create an early stopping monitor.

        :param rel_patience: *Relative* patience value. Limits the training to last at most `(1 + rel_patience)` times as much steps as it took to reach best val loss value.
        :param margin: Consider val loss to have improved if the *relative* improvement is at least as great as this.
        :param min_steps: Minimum number of steps for the training process.
        :param max_steps: Maximum number of steps for the training process."""

        self.rel_patience = rel_patience
        self.margin = margin
        self.min_steps = min_steps
        self.max_steps = max_steps
        self._best_v, self._best_t = None, None

    def __call__(self, value: float, opt_step: int):
        if self.max_steps is not None:
            if opt_step > self.max_steps:
                return True

        if self._best_v is None:
            self._best_v, self._best_t = value, opt_step
            return False
        else:
            improv = (self._best_v - value) / value
            if improv < self.margin:
                patience = self.rel_patience * opt_step / (1.0 + self.rel_patience)
                return opt_step - self._best_t > patience and opt_step > self.min_steps
            else:
                self._best_v, self._best_t = value, opt_step
                return False
