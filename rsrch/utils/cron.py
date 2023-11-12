class Flag:
    def __bool__(self):
        ...


class Every:
    def __init__(self, step_fn, period: int, hook=None):
        self.step_fn = step_fn
        self.period = period
        self.hook = hook
        self._cur, self._last, self._ret = None, None, True

    def __bool__(self):
        cur_step = self.step_fn()
        if self._cur != cur_step:
            self._ret = self._last is None or cur_step - self._last >= self.period
            if self._ret:
                self._last = cur_step
            self._cur = cur_step
        return self._ret

    def __call__(self, *args, **kwargs):
        if self.hook is not None:
            if self:
                self.hook(*args, **kwargs)


class Once:
    def __init__(self, cond, hook=None):
        self.cond = cond
        self.hook = hook
        self._done = False

    def __bool__(self):
        if not self._done:
            self._done = self.cond()
        return self._done

    def __call__(self, *args, **kwargs):
        if self.hook is None:
            return
        if self:
            return self.hook(*args, **kwargs)


class Never:
    def __bool__(self):
        return False

    def __call__(self, *args, **kwargs):
        pass
