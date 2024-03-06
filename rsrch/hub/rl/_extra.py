class MaxNorm1d(nn.Module):
    def __init__(
        self, in_features: int, affine: bool = True, track_running_stats: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.affine = affine
        self.track_running_stats = track_running_stats
        if track_running_stats:
            _running_max = torch.empty(in_features)
            _running_max.fill_(-torch.inf)
            self.register_buffer("_running_max", _running_max)
            if affine:
                _running_min = torch.empty(in_features)
                _running_min.fill_(torch.inf)
                self.register_buffer("_running_min", _running_min)

    def forward(self, x: Tensor) -> Tensor:
        if self.track_running_stats:
            if self.training:
                x_ = x.detach()
                if self.affine:
                    x_max, x_min = x_.amax(0), x_.amin(0)
                    self._running_max = torch.max(self._running_max, x_max)
                    self._running_min = torch.min(self._running_min, x_min)
                else:
                    x_max = x_.abs().amax(0)
                    self._running_max = torch.max(self._running_max, x_max)

        if self.affine:
            if self.track_running_stats:
                x_min, x_max = self._running_min, self._running_max
            else:
                x_ = x.detach()
                x_min, x_max = x_.amin(0), x_.amax(0)
            loc, scale = 0.5 * (x_min + x_max), 0.5 * (x_max - x_min)
            x = (x - loc) / scale
        else:
            if self.track_running_stats:
                x_max = self._running_max
            else:
                x_ = x.detach()
                x_max = x_.abs().amax(0)
            x = x / x_max

        return x


class Scaler1d(nn.Module):
    def __init__(self, loc=None, scale=None):
        super().__init__()
        if loc is None:
            loc = 0.0
        self.register_buffer("loc", torch.as_tensor(loc))
        if scale is None:
            scale = 1.0
        self.register_buffer("scale", torch.as_tensor(scale))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.loc) / self.scale
