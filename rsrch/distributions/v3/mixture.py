from typing import List

from .distribution import Distribution
from .tensorlike import Tensorlike


class Mixture(Distribution, Tensorlike):
    def __init__(self, index: Distribution, values: List[Distribution]):
        batch_shape = index.batch_shape
        Tensorlike.__init__(self, batch_shape)
        self.event_shape = values[0].event_shape

        self.index: Distribution
        self.register_field("index", index)

        self.values: List[Distribution]
        for val_idx in range(len(values)):
            self.register_field(f"value{val_idx}", values[0])
