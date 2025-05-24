from torch import Tensor, nn


class NormLayer(nn.Module):
    """A 'protocol class' for norm layers."""

    def __init__(self, num_features: int):
        ...


class ActLayer(nn.Module):
    """A 'protocol class' for activation layers."""

    def __init__(self, inplace: bool):
        ...
