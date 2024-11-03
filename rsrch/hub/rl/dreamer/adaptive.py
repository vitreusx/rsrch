import math
from collections import deque
from typing import Literal

import numpy as np
from scipy.stats import norm


def _grid_of_values(value_range: tuple[float, float], gap: float):
    min_value, max_value = value_range
    num_values = (math.log(max_value) - math.log(min_value)) / math.log(gap)
    num_values = int(num_values) + 1
    return np.geomspace(min_value, max_value, num_values)


class V1:
    def __init__(
        self,
        ratio_range: tuple[float, float],
        update_mult: float,
        initial_value: float | None = None,
        variant: Literal["0", "1", "2"] = "0",
    ):
        self.variant = variant

        self.values = _grid_of_values(ratio_range, update_mult)
        assert np.all(np.diff(self.values) >= 0)
        if initial_value is not None:
            self.index = min(
                range(len(self.values)),
                key=lambda idx: np.abs(self.values[idx] - initial_value),
            )
        else:
            self.index = len(self.values) // 2

        self._prev_loss, self._prev_time = None, None
        self._prev_index, self.vel = None, None

    def update(self, loss: float, time: float):
        metrics = {}
        if self._prev_loss is not None:
            index = self.index
            vel = (loss - self._prev_loss) / (time - self._prev_time)
            metrics["loss_vel"] = vel
            if self.vel is None:
                self._try_decrement()
            elif self.variant == "0":
                if vel < 0:
                    self._try_decrement()
                else:
                    self._try_increment()
            elif vel > 0 and self.variant == "2":
                self._try_increment()
            else:
                accel = (vel - self.vel) / (self.index - self._prev_index)
                if accel < 0:
                    self._try_increment()
                else:
                    self._try_decrement()
            self.vel = vel
            self._prev_index = index
        self._prev_loss = loss
        self._prev_time = time
        return metrics

    def _try_decrement(self):
        if self.index > 0:
            self.index -= 1
        else:
            self.index += 1

    def _try_increment(self):
        if self.index < len(self.values) - 1:
            self.index += 1
        else:
            self.index -= 1

    @property
    def value(self):
        return self.values[self.index]


class V2:
    def __init__(
        self,
        ratio_range: tuple[float, float],
        update_mult: float,
        variant: Literal["0"] = "0",
        initial_value: float | None = None,
        try_unused_after: float = 50e3,
    ):
        self.variant = variant
        self.try_unused_after = try_unused_after

        self.values = _grid_of_values(ratio_range, update_mult)
        assert np.all(np.diff(self.values) >= 0)
        if initial_value is not None:
            self.index = min(
                range(len(self.values)),
                key=lambda idx: np.abs(self.values[idx] - initial_value),
            )
        else:
            self.index = len(self.values) // 2

        self._prev_loss, self._prev_time = None, None
        self.velocities = [None for _ in range(len(self.values))]
        self.last_updated = [None for _ in range(len(self.values))]
        self.index = range(len(self.values))[0]

    def update(self, loss: float, time: float):
        metrics = {}

        if self._prev_loss is not None:
            vel = (loss - self._prev_loss) / (time - self._prev_time)
            metrics["loss_vel"] = vel
            self.velocities[self.index] = vel
            self.last_updated[self.index] = time
            self.index = self._next_index(time)

        self._prev_loss = loss
        self._prev_time = time

        return metrics

    def _next_index(self, time: float):
        for idx in range(len(self.values)):
            last_updated = self.last_updated[idx]
            if last_updated is None or time - last_updated >= self.try_unused_after:
                return idx

        return min(
            range(len(self.values)),
            key=lambda idx: self.velocities[idx],
        )

    @property
    def value(self):
        return self.values[self.index]


class V3:
    def __init__(
        self,
        ratio_range: tuple[float, float],
        update_mult: float,
        variant: Literal["0", "1"] = "0",
        alpha: float = 0.05,
        window: float = 25e3,
    ):
        self.values = _grid_of_values(ratio_range, update_mult)
        self.index = 0
        self.alpha = alpha
        self.window = window
        self.history = deque()

    def update(self, loss: float, time: float):
        metrics = {}

        self.history.append((time, loss))
        while True:
            prev_time, _ = self.history[0]
            if time - prev_time >= self.window:
                self.history.popleft()
            else:
                break

        next_index = self.index
        if len(self.history) > 10:
            values = np.array([loss for _, loss in self.history])
            # Mann-Kendall test
            S = np.tril(np.subtract.outer(values, values)).sum()
            n = len(values)
            V = 1 / 18.0 * (n * (n - 1) * (2 * n + 5))  # Assuming no ties
            Z = (S - np.sign(S)) / np.sqrt(V)
            p = norm.cdf(Z)
            if p < self.alpha:
                next_index = max(self.index - 1, 0)
            elif p > 1.0 - self.alpha:
                next_index = min(self.index + 1, len(self.values) - 1)

        if next_index != self.index:
            self.index = next_index
            self.history.clear()

        return metrics

    @property
    def value(self):
        return self.values[self.index]


class V4:
    NO_VAL_LOSS = True

    def __init__(
        self,
        init_ratio: float,
        init_time: float,
        final_ratio: float,
        final_time: float,
    ):
        t0, r0, t1, r1 = init_time, init_ratio, final_time, final_ratio
        self.gamma = (math.log(r1) - math.log(r0)) / (math.log(t1) - math.log(t0))
        self.log_A = math.log(r1) - self.gamma * math.log(t1)
        self.time = init_time

    def update(self, loss: float, time: float):
        self.time = time
        return {}

    @property
    def value(self):
        return math.exp(self.gamma * math.log(self.time) + self.log_A)


__all__ = [
    "V1",
    "V2",
    "V3",
    "V4",
]
