import math
from numbers import Number

import numpy as np


class Constant:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value


class Linear:
    def __init__(self, init, final, over):
        self.init = init
        self.final = final
        self.over = over

    def __call__(self, t):
        t = min(t / self.over, 1.0)
        return self.init * (1.0 - t) + self.final * t


class Exp:
    def __init__(self, init, final, rate=None, half_life=None):
        assert (rate is not None) ^ (half_life is not None)
        if half_life is not None:
            rate = math.log(2) / half_life
        self.init = init
        self.final = final
        self.rate = rate

    def __call__(self, t):
        return self.final + (self.init - self.final) * math.exp(-self.rate * t)


class Piecewise:
    def __init__(self, *args):
        self.values, self.pivots = [*args[::2]], np.asarray(args[1::2])
        for idx, val in enumerate(self.values):
            if isinstance(val, Number):
                self.values[idx] = Constant(val)

    def __call__(self, t):
        idx = np.searchsorted(self.pivots, t)
        return self.values[idx](t)


class Auto:
    def __init__(self, desc, step_fn):
        classes = [Constant, Linear, Exp, Piecewise]
        locals = {cls.__name__: cls for cls in classes}

        if isinstance(desc, str):
            self._sched = eval(desc, globals(), locals)
        else:
            self._sched = Constant(desc)

        self.step_fn = step_fn

    def __call__(self):
        return self._sched(self.step_fn())
