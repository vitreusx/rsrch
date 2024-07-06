from rsrch.utils import cron


class EarlyStopping:
    def __init__(self, patience: int, margin: float = 0.0, maximize=True):
        self.patience = patience
        self.margin = margin
        self.maximize = maximize
        self._best, self._streak = None, 0

    def __call__(self, value):
        if self._best is None:
            self._best, self._streak = value, 0
            return False
        else:
            improv = (self._best - value) / self._best
            if self.maximize:
                improv = -improv

            if improv < self.margin:
                self._streak += 1
                return self._streak < self.patience
            else:
                self._best, self._streak = value, 0
                return False


class Trainer:
    def train_step(self, batch):
        ...

    def val_epoch(self, loader):
        ...
