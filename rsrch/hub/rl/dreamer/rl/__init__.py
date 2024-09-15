from torch import Tensor

import rsrch.distributions as D


class Actor:
    def __call__(self, state: Tensor) -> D.Distribution:
        ...
