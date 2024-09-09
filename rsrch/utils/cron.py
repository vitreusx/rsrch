from typing import Callable


class Every:
    """A flag for running actions periodically.

    `__bool__` returns `True` on first call, and whenever the step value (given by `step_fn()`) has increased by at least `every` since the last time `__bool__` returned `True`. When calling `__bool__` multiple times when the step value hasn't changed:

    - if `iters` is None, always return `True`;
    - otherwise, if `accumulate`, return `True` for as long as the ratio of the total # of `True`s returned and the step value is lower than `iters`/`every`. So we accumulate unused iterations from the past.
    - otherwise, if not `accumulate`, return `True` at most `iters` times - in other words, previous unused iterations are lost.
    """

    def __init__(
        self,
        step_fn: Callable[[], float],
        period: float,
        iters: int | None = 1,
        accumulate: bool = False,
    ):
        self.step_fn = step_fn
        self.period = period
        self.iters = iters
        self.accumulate = accumulate
        self.reset()

    def reset(self):
        self._last = None
        self._acc = 0

    def __bool__(self):
        step = self.step_fn()
        if self._last is None or step - self._last >= self.period:
            if self.accumulate and self._last is not None:
                self._acc += self.iters * (step - self._last) / self.period
            else:
                self._acc = self.iters
            self._last = step

        if step == self._last:
            if self._acc is None:
                return True
            elif self._acc > 0:
                self._acc -= 1
                return True

        return False


class Until:
    def __init__(self, step_fn, max_value):
        self.step_fn = step_fn
        self.max_value = max_value

    def __bool__(self):
        return self.step_fn() <= self.max_value


class Never:
    def __bool__(self):
        return False
