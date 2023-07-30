import numpy as np
import torch


class Stats:
    def __init__(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.values = [*values]
        self._update()

    def _update(self):
        self.count = len(self.values)
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        self.min = np.min(self.values)
        self.q25 = np.quantile(self.values, q=0.25)
        self.med = np.median(self.values)
        self.q75 = np.quantile(self.values, q=0.75)
        self.max = np.max(self.values)

    def append(self, value):
        self.values.append(value)
        self._update()
        return self

    def extend(self, values):
        self.values.extend(values)
        self._update()
        return self

    def asdict(self):
        KEYS = ["min", "q25", "mean", "med", "q75", "max"]
        return {k: getattr(self, k) for k in KEYS}


class RunningStats:
    def __init__(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        self.count = len(values)
        self.mean = values.mean()
        self.M2 = ((values - self.mean) ** 2).sum()
        self.min, self.max = values.min(), values.max()

    def extend(self, other):
        for val in other:
            self.append(val)
        return self

    def append(self, value):
        self.count += 1
        delta1 = value - self.mean
        self.mean += delta1 / self.count
        delta2 = value - self.mean
        self.M2 += delta1 * delta2
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        return self

    @property
    def var(self):
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return np.sqrt(self.var)

    def asdict(self):
        return dict(min=self.min, mean=self.mean, max=self.max)
