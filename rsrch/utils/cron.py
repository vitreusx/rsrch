class Every:
    """A flag for running actions based on a loop counter, or other
    monotonically increasing variable."""

    def __init__(
        self,
        step_fn,
        every: int | None = 1,
        iters: int | None = 1,
        never: bool = False,
    ):
        """Create the flag variable.
        :param step_fn: Step function, i.e. a function which returns current step value/loop counter.
        :param every: How often (in terms of steps) to signal True.
        :param iters: How many times, once the period of `every` has passed, to signal True.
        """

        self.step_fn = step_fn
        if every is None or iters is None or iters == 0:
            never = True
        self.every, self.iters, self.never = every, iters, never
        self._last, self._ret = None, True
        self._acc = 0

    def __bool__(self):
        if self.never:
            return False

        cur_step = self.step_fn()

        if self._last is None:
            # On the first iteration, do a "full" cycle.
            self._acc += self.iters
            self._last = cur_step
        else:
            while cur_step - self._last >= self.every:
                self._last += self.every
                self._acc += self.iters

        if self._acc > 0:
            self._acc -= 1
            return True
        else:
            return False


class Until:
    def __init__(self, step_fn, max_value):
        self.step_fn = step_fn
        self.max_value = max_value

    def __bool__(self):
        return self.step_fn() <= self.max_value
