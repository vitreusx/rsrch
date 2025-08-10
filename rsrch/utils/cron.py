from typing import Callable


class Flag:
    """Flag class."""

    def __bool__(self): ...


class Every(Flag):
    """A flag for running actions periodically.

    Formally, the flag itself doesn't perform actions, but rather provides a way
    to check (via bool conversion) if the action should be performed, and for how long.

    The flag is parametrized by:

    - `step_fn`: a lambda used to get current time/step value;
    - `period`: the value of the time period between firing the flag;
    - `iters`: the number of times to run the action.

    There are two modes of operation, controlled by `accumulate` parameter:

    - `accumulate = True`: Every `period` steps, starting from the first call,
    `iters`number of actions are added to an accumulator. `bool(flag)` returns
    `True` as long as the number of leftover actions is greater than zero.
    - `accumulate = False`: At the point of first check `bool(flag)`, it returns
    `True` at most `iters` number of times, as long as the step value remains
    the same. Thereafter, the flag activates once again only after at least
    `period` steps have elapsed since the last time it was active, and again
    remains active for `iters` number of steps for as long as the step value
    remains the same.

    Usage and examples are motivated mostly by RL applications, where step
    values are nontrivial compared to supervised learning:

    - If you want to perform optimization step every `K` environment steps on
    average, you want to use `accumulate = True`.
    - If you want to write logs or save stats to the dashboard every so often,
    accumulation is unnecessary, so you'd rather use `accumulate = False` with
    `iters = 1`.
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
        if self.period == 0 or self.iters == 0:
            return False

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


class Until(Flag):
    """A flag for performing action until a given step value."""

    def __init__(
        self,
        step_fn: Callable[[], float],
        max_value: float,
    ):
        self.step_fn = step_fn
        self.max_value = max_value

    def __bool__(self):
        return self.step_fn() <= self.max_value


class Never(Flag):
    """A flag for never performing an action."""

    def __bool__(self):
        return False


class Always(Flag):
    """A flag for always performing an action."""

    def __bool__(self):
        return True


class Once(Flag):
    def __init__(self):
        self._fired = False

    def __bool__(self):
        ret = not self._fired
        self._fired = True
        return ret
