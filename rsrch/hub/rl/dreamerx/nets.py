import numpy as np
from torch import Tensor, nn

import rsrch.distributions as D
from rsrch import spaces
from rsrch.nn import dist_head as dh
from rsrch.nn import fc


class VisEncoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Image,
        conv_hidden: int,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        assert (space.width, space.height) == (64, 64)
        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        layers = [
            nn.Conv2d(space.shape[0], conv_hidden, 4, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.Conv2d(conv_hidden, 2 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.Conv2d(2 * conv_hidden, 4 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.Conv2d(4 * conv_hidden, 8 * conv_hidden, 4, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            # At this point the size is [2, 2]
            nn.Flatten(),
        ]

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        super().__init__(*layers)


class VisDecoder(nn.Module):
    def __init__(
        self,
        space: spaces.torch.Image,
        state_dim: int,
        conv_hidden: int,
        out_values: int = 1,
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__()
        self.out_values = out_values
        assert tuple(space.shape[-2:]) == (64, 64)

        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        self.fc = nn.Linear(state_dim, 8 * conv_hidden)

        layers = [
            nn.ConvTranspose2d(8 * conv_hidden, 4 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(4 * conv_hidden),
            nn.ConvTranspose2d(4 * conv_hidden, 2 * conv_hidden, 5, 2),
            act_layer(),
            norm_layer(2 * conv_hidden),
            nn.ConvTranspose2d(2 * conv_hidden, conv_hidden, 6, 2),
            act_layer(),
            norm_layer(conv_hidden),
            nn.ConvTranspose2d(conv_hidden, out_values * space.shape[0], 6, 2),
        ]

        layers = [x for x in layers if not isinstance(x, nn.Identity)]
        self.convt = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 1, 1, x.shape[1])
        x = self.convt(x)
        x = x.reshape(len(x), self.out_values, -1, *x.shape[2:])
        return x


class Reshape(nn.Module):
    def __init__(self, shape: tuple[int, ...], start_dim=1, end_dim=-1):
        super().__init__()
        self.shape = shape
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        new_shape = x.shape[: self.start_dim] + self.shape + x.shape[self.end_dim :][1:]
        return x.reshape(new_shape)


class ProprioEncoder(Reshape):
    def __init__(self, space: spaces.torch.Box):
        super().__init__([])


class ProprioDecoder(nn.Sequential):
    def __init__(
        self,
        space: spaces.torch.Box,
        state_dim: int,
        fc_layers: list[int],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        out_dim = int(np.prod(space.shape))
        super().__init__(
            fc.FullyConnected(
                layer_sizes=[state_dim, *fc_layers, out_dim],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            Reshape(space.shape),
        )


class RewardPred(nn.Sequential):
    def __init__(
        self,
        state_dim: int,
        fc_layers: list[int],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [state_dim, *fc_layers, 1],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            Reshape([]),
        )


class TermPred(nn.Sequential):
    def __init__(
        self,
        state_dim: int,
        fc_layers: list[int],
        norm_layer=None,
        act_layer=nn.ELU,
    ):
        super().__init__(
            fc.FullyConnected(
                [state_dim, *fc_layers, 1],
                norm_layer=norm_layer,
                act_layer=act_layer,
                final_layer="act",
            ),
            dh.Bernoulli(fc_layers[-1]),
        )
