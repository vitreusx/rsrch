from collections import deque


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
