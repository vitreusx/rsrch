from typing import Callable


class Every:
    """A flag for running actions periodically.

    The behavior is as follows. Consider following code block:

    ```python
    step = step_0
    should_do = Every(lambda: step, period, iters)
    while True:
        while should_do:
            perform_action()
        step += S
    ```

    Then,
    - If `iters` is None, `bool(should_do)` is True every `period` steps, and stays True for as long as the step value is the same (in the case of the code above, `while` loop wouldn't terminate.) This setting is useful for e.g. logging metrics every so often, since for a given time step we want to log all the metrics, i.e. `bool(...)` should be True for the entire step.
    - If `iters` is not None, then:
        - `while` loop may repeat zero or more times;
        - At step value `step`, we expect `perform_action` to have been invoked roughly `(step - step_0) * iters / period` times: every `period` steps, `while` loop will get repeated `iters` times.
        - The step increment `S` may be greater than 1 - the mean number of calls to `perform_action` per single step will be same as above.
    """

    def __init__(
        self,
        step_fn: Callable[[], float],
        every: float,
        iters: int | None = 1,
        never: bool = False,
    ):
        self.step_fn = step_fn
        self.every = every
        self.iters = iters
        self.never = never
        self.reset()

    def reset(self):
        self._last = None
        self._acc = 0

    def __bool__(self):
        if self.never:
            return False

        step = self.step_fn()
        if self.iters is None:
            if (self._last is None) or (step - self._last >= self.every):
                self._last = step
            return step == self._last
        else:
            if self._last is None:
                self._acc = self.every * self.iters
            else:
                self._acc += (step - self._last) * self.iters
            self._last = step
            if self._acc >= self.every:
                self._acc -= self.every
                return True
            else:
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
