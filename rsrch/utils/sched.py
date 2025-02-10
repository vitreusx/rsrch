import math
from numbers import Number
from typing import Callable

import numpy as np


class Constant:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, t):
        return self.value


class Linear:
    def __init__(self, *points: tuple[float, float]):
        points = np.array(points)
        self.ts, self.vs = points[:, 0], points[:, 1]

    def __call__(self, t: float):
        return float(np.interp(t, self.ts, self.vs))


class LogLinear:
    def __init__(self, *points: tuple[float, float]):
        points = np.array(points)
        self.ts, self.log_vs = points[:, 0], np.log(points[:, 1])

    def __call__(self, t: float):
        log_v = np.interp(t, self.ts, self.log_vs)
        return math.exp(log_v)


class Exp:
    def __init__(
        self,
        init: tuple[float, float],
        final: tuple[float, float],
        half_time: float,
    ):
        self.t0, self.v0 = init
        self.t1, self.v1 = final
        self.lmbd = -math.log(2) / half_time
        self.A = (self.v1 - self.v0) / (math.exp(self.lmbd * (self.t1 - self.t0)) - 1)
        self.b = self.v0 - self.A

    def __call__(self, t: float):
        t = max(min(t, self.t1), self.t0)
        return self.A * math.exp(self.lmbd * (t - self.t0)) + self.b


class LogPoly:
    def __init__(self, *pts: tuple[float, float], deg: int | None = None):
        pts = np.array(pts)
        x, log_y = pts[:, 0], np.log(pts[:, 1])
        if deg is None:
            deg = len(x) - 1
        self.poly = np.polynomial.Polynomial.fit(x, log_y, deg)

    def __call__(self, t: float):
        return math.exp(self.poly(t))


class Piecewise:
    def __init__(self, *args: float):
        self.values, self.pivots = [*args[::2]], np.asarray(args[1::2])
        for idx, val in enumerate(self.values):
            if isinstance(val, Number):
                self.values[idx] = Constant(val)

    def __call__(self, t):
        idx = np.searchsorted(self.pivots, t)
        return self.values[idx](t)


class Auto:
    def __init__(
        self,
        desc: str | float,
        step_fn: Callable[[], float],
    ):
        classes = [Constant, LogLinear, Linear, Exp, Piecewise]
        locals = {cls.__name__.lower(): cls for cls in classes}

        if isinstance(desc, str):
            self._sched = eval(desc, globals(), locals)
        else:
            self._sched = Constant(desc)

        self.step_fn = step_fn

    def __call__(self):
        return self._sched(self.step_fn())
