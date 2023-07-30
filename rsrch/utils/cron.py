class Every:
    def __init__(self, step_fn, period: int, hook=None):
        self.step_fn = step_fn
        self.period = period
        self.hook = hook
        self._step, self._ret = None, True

    def __bool__(self):
        cur_step = self.step_fn()
        if self._step != cur_step:
            self._ret = self._step is None or cur_step - self._step >= self.period
            self._step = cur_step
        return self._ret

    def __call__(self, *args, **kwargs):
        if self.hook is not None:
            if self:
                self.hook(*args, **kwargs)
