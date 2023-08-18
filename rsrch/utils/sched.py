import math


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


class schedule:
    def __init__(self, desc: str):
        const = Constant
        linear = Linear
        exp = Exp
        self._sched = eval(desc)

    def __call__(self, t):
        return self._sched(t)
