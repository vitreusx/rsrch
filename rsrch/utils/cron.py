class Every:
    def __init__(self, step_fn, period: int | None = None):
        self.step_fn = step_fn
        self.period = period
        self._cur, self._last, self._ret = None, None, True

    def __bool__(self):
        if self.period is None:
            return False

        cur_step = self.step_fn()
        if self._cur != cur_step:
            self._ret = self._last is None or cur_step - self._last >= self.period
            if self._ret:
                self._last = cur_step
            self._cur = cur_step
        return self._ret


class Every2:
    """A flag for running actions based on a loop counter, or other
    monotonically increasing variable."""

    def __init__(self, step_fn, every=1, iters=1, never=False):
        """Create the flag variable.
        :param step_fn: Step function, i.e. a function which returns current step value/loop counter.
        :param every: How often (in terms of steps) to signal True.
        :param iters: How many times, once the period of `every` has passed, to signal True.
        """

        self.step_fn = step_fn
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


class Every3:
    """A flag for running actions based on a loop counter, or other
    monotonically increasing variable."""

    def __init__(self, step_fn, iter_fn, every=1, iters=1, never=False):
        """Create the flag variable.
        :param step_fn: Step function, i.e. target loop counter.
        :param iter_fn: Iter count function, i.e. execution loop counter.
        :param every: How often (in terms of steps) to signal True.
        :param iters: How many times, once the period of `every` has passed, to signal True.
        """

        self.step_fn, self.iter_fn = step_fn, iter_fn
        self._step0, self._iter0 = self.step_fn(), self.iter_fn()
        self.every, self.iters = every, iters

    def __bool__(self):
        cur_step = self.step_fn() - self._step0
        cur_iter = self.iter_fn() - self._iter0
        return cur_step * self.iters > cur_iter * self.every


class Once:
    def __init__(self, cond):
        self.cond = cond
        self._done = False

    def __bool__(self):
        if not self._done:
            self._done = self.cond()
        return self._done


class Never:
    def __bool__(self):
        return False
