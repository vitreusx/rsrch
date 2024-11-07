import math
from numbers import Number
from typing import Callable

import numpy as np


class Constant:
    def __init__(self, value):
        self.value = value

    def __call__(self, t):
        return self.value


class Linear:
    def __init__(self, t0, v0, t1, v1):
        self.t0 = t0
        self.v0 = v0
        self.t1 = t1
        self.v1 = v1

    def __call__(self, t):
        t = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return self.v0 * (1.0 - t) + self.v1 * t


class Exp:
    def __init__(self, init, final, rate=None, half_life=None):
        assert (rate is not None) ^ (half_life is not None)
        if half_life is not None:
            rate = math.log(2) / half_life
        self.init = init
        self.final = final
        self.rate = rate

    def __call__(self, t):
        return self.final + (self.init - self.final) * np.exp(-self.rate * t)


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
        locals = {cls.__name__.lower(): cls for cls in classes}

        if isinstance(desc, str):
            self._sched = eval(desc, globals(), locals)
        else:
            self._sched = Constant(desc)

        self.step_fn = step_fn

    def __call__(self):
        return self._sched(self.step_fn())
