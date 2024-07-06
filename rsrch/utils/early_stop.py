class EarlyStopping:
    def __init__(
        self,
        patience: int,
        margin: float = 0.0,
        min_steps: int = 0,
        maximize=False,
    ):
        self.patience = patience
        self.margin = margin
        self.min_steps = min_steps
        self.maximize = maximize
        self._best_v, self._best_t = None, None

    def __call__(self, value: float, opt_step: int):
        if opt_step < self.min_steps:
            return False

        if self._best_v is None:
            self._best_v, self._best_t = value, opt_step
            return False
        else:
            improv = (self._best_v - value) / self._best_v
            if self.maximize:
                improv = -improv

            if improv < self.margin:
                return opt_step - self._best_t > self.patience
            else:
                self._best_v, self._best_t = value, opt_step
                return False
