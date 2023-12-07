from collections import deque

import numpy as np


class RunningAvg:
    def __init__(self, last_k=None, decay=None):
        self.value = None
        assert last_k is not None or decay is not None
        self._last_k = deque(maxlen=last_k) if last_k is not None else None
        self._decay = decay

    def update(self, x):
        if self._last_k is not None:
            self._last_k.append(x)
            self.value = sum(self._last_k) / len(self._last_k)
        else:
            if self.value is None:
                self.value = x
            else:
                self.value = self._decay * self.value + (1.0 - self._decay) * x


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    @property
    def std(self):
        return np.sqrt(self.var)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
