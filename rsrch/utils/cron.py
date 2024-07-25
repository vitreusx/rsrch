class Every:
    """A flag for running actions based on a loop counter, or other
    monotonically increasing variable."""

    def __init__(
        self,
        step_fn,
        every: int = 1,
        iters: int | None = 1,
        never: bool = False,
    ):
        """Create the flag variable.
        :param step_fn: Step function, i.e. a function which returns current step value/loop counter.
        :param every: How often (in terms of steps) to signal True.
        :param iters: How many times, once the period of `every` has passed, to signal True. If None, the flag is on
        """

        self.step_fn = step_fn
        self.every, self.iters, self.never = every, iters, never
        self._last, self._sent, self._acc = None, None, None

    def __bool__(self):
        if self.never:
            return False

        cur_step = self.step_fn()

        if self._last is None:
            self._last = self._sent = cur_step
            if self.iters is not None:
                self._acc = self.iters
        elif cur_step - self._sent >= self.every:
            cycles = (cur_step - self._sent) // self.every
            self._sent += cycles * self.every
            self._last = cur_step
            if self.iters is not None:
                self._acc = self.iters

        if cur_step == self._last:
            if self._acc is None:
                return True
            elif self._acc >= 1:
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
