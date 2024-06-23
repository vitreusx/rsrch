import math
from copy import copy
from typing import Callable, Iterable, ParamSpec, TypeVar, cast

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization

from rsrch import spaces


class Reshape(nn.Module):
    def __init__(
        self,
        in_shape: int | tuple[int, ...],
        out_shape: int | tuple[int, ...],
    ):
        super().__init__()
        if not isinstance(in_shape, Iterable):
            in_shape = (in_shape,)
        self.in_shape = tuple(in_shape)
        if not isinstance(out_shape, Iterable):
            out_shape = (out_shape,)
        self.out_shape = tuple(out_shape)

    def forward(self, input: Tensor) -> Tensor:
        new_shape = [*input.shape[: -len(self.in_shape)], *self.out_shape]
        return input.reshape(new_shape)


class Normalize(nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.first_time = True
        self.loc: Tensor
        self.register_buffer("loc", torch.zeros([]))
        self.inv_scale: Tensor
        self.register_buffer("inv_scale", torch.ones([]))

    def forward(self, input: Tensor):
        if self.training and self.first_time:
            self.loc.copy_(input.mean())
            self.inv_scale.copy_(input.std().clamp_min(1e-8).reciprocal())
            self.first_time = False
        input = (input - self.loc) * self.inv_scale
        return input

    def inverse(self, input: Tensor):
        return input / self.inv_scale + self.loc


class ResBlock(nn.Module):
    def __init__(self, num_features: int, norm_layer, act_layer):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            *([norm_layer(num_features)] if norm_layer else []),
            act_layer(),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            *([norm_layer(num_features)] if norm_layer else []),
        )
        self.act = act_layer()

    def forward(self, input: Tensor):
        return self.act(input + self.net(input))


class Encoder_v1(nn.Sequential):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        conv_hidden: int,
        *,
        scale_factor: int = 16,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
    ):
        assert obs_space.width % scale_factor == 0
        assert obs_space.height % scale_factor == 0

        num_stages = int(math.log2(scale_factor))
        assert 2**num_stages == scale_factor
        channels = [obs_space.num_channels]
        channels.extend(2**stage * conv_hidden for stage in range(num_stages))

        layers = []
        for idx, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            if idx > 0:
                if norm_layer is not None:
                    layers.append(norm_layer(in_ch))
                layers.append(act_layer())
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1))

        for _ in range(res_blocks):
            layers.append(ResBlock(channels[-1], norm_layer, act_layer))

        super().__init__(*layers)


class Decoder_v1(nn.Module):
    def __init__(
        self,
        obs_space: spaces.torch.Image,
        conv_hidden: int,
        *,
        scale_factor: int = 16,
        norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU,
        res_blocks: int = 2,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = lambda _: nn.Identity()

        assert obs_space.width % scale_factor == 0
        cur_w = obs_space.width // scale_factor

        assert obs_space.height % scale_factor == 0
        cur_h = obs_space.height // scale_factor

        cur_nc = conv_hidden * (scale_factor // 2)

        self.in_shape = (cur_nc, cur_h, cur_w)

        layers = [
            ResBlock(conv_hidden, norm_layer, act_layer),
            ResBlock(conv_hidden, norm_layer, act_layer),
        ]

        num_stages = int(math.log2(scale_factor))
        assert scale_factor == 2**num_stages
        channels = [2**stage * conv_hidden for stage in reversed(range(num_stages))]
        channels += [obs_space.num_channels]

        layers = []
        for _ in range(res_blocks):
            layers.append(ResBlock(channels[0], norm_layer, act_layer))

        for stage, (in_ch, out_ch) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
            if stage < num_stages - 1:
                if norm_layer is not None:
                    layers.append(norm_layer(out_ch))
                layers.append(act_layer())

        self.main = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        return self.main(input)


class LayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        elementwise_affine=True,
        bias=True,
    ):
        super().__init__()
        self._norm = nn.LayerNorm(
            normalized_shape=(num_features,),
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(self, input: Tensor):
        input = input.moveaxis(-3, -1)
        input = self._norm(input)
        input = input.moveaxis(-1, -3)
        return input


class AtariEncoder(nn.Sequential):
    def __init__(self, hidden=48, in_channels=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = lambda _: None

        layers = [
            nn.Conv2d(in_channels, hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(hidden, 2 * hidden, 4, 2),
            nn.ELU(),
            nn.Conv2d(2 * hidden, 4 * hidden, 4, 2),
            norm_layer(4 * hidden),
            nn.ELU(),
            nn.Conv2d(4 * hidden, 8 * hidden, 4, 2),
            norm_layer(8 * hidden),
            nn.ELU(),
            nn.Flatten(),
        ]

        layers = [x for x in layers if x is not None]

        super().__init__(*layers)


class Lipschitz(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = P.spectral_norm(net)
        # device = next(self.net.parameters()).device
        # log_K0 = np.log(np.e - 1)  # F.softplus(log_K0) = 1
        # self.log_K = nn.Parameter(log_K0 * torch.ones((), device=device))

    def forward(self, input: Tensor):
        return self.net(input)
        # return self.net(input) * F.softplus(self.log_K)


class AtariDecoder(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden=48,
        out_channels=1,
        bounded_lipschitz: bool = False,
    ):
        layers = [
            nn.Linear(in_features, 32 * hidden),
            nn.ELU(),
            Reshape(32 * hidden, (32 * hidden, 1, 1)),
            nn.ConvTranspose2d(32 * hidden, 4 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(4 * hidden, 2 * hidden, 5, 2),
            nn.ELU(),
            nn.ConvTranspose2d(2 * hidden, hidden, 6, 2),
            nn.ELU(),
            nn.ConvTranspose2d(hidden, out_channels, 6, 2),
        ]

        if bounded_lipschitz:
            for idx, layer in enumerate(layers):
                if isinstance(layer, (nn.Linear, nn.ConvTranspose2d)):
                    layer = Lipschitz(layer)
                layers[idx] = layer

        super().__init__(*layers)


P_ = ParamSpec("P_")
T = TypeVar("T")


def copy_sig(source: Callable[P_, T]) -> Callable[[Callable[..., T]], Callable[P_, T]]:
    def wrapper(func: Callable[..., T]) -> Callable[P_, T]:
        func = copy(cast(Callable[P_, T], func))
        func.__doc__ = source.__doc__
        return func

    return wrapper


def softplus_inv(y: Tensor, beta=1.0, threshold=20.0):
    thresh_inv = F.softplus(
        torch.tensor(threshold / beta).type_as(y),
        beta=beta,
        threshold=threshold,
    )
    return torch.where(
        y < thresh_inv,
        torch.expm1(beta * y).log() / beta,
        y,
    )


class SNLinear(nn.Linear):
    @copy_sig(nn.Linear.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_parametrization(
            self,
            "weight",
            P._SpectralNorm(self.weight, dim=0),
        )
        scale = (self.parametrizations.weight.original / self.weight).nanmean()
        self._log_scale = nn.Parameter(softplus_inv(scale, beta=1.0e5))

    def forward(self, input: Tensor):
        scale = F.softplus(self._log_scale, beta=1.0e5)
        output = F.linear(
            input=input,
            weight=self.weight * scale,
            bias=self.bias,
        )
        return output, scale


class SNConv2d(nn.Conv2d):
    @copy_sig(nn.Conv2d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_parametrization(
            self,
            "weight",
            P._SpectralNorm(self.weight, dim=0),
        )
        scale = (self.parametrizations.weight.original / self.weight).nanmean()
        self._log_scale = nn.Parameter(softplus_inv(scale, beta=1.0e5))

    def forward(self, input: Tensor):
        scale = F.softplus(self._log_scale, beta=1.0e5)
        output = F.conv2d(
            input=input,
            weight=self.weight * scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output, scale


class SNConvTranspose2d(nn.ConvTranspose2d):
    @copy_sig(nn.ConvTranspose2d.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_parametrization(
            self,
            "weight",
            P._SpectralNorm(self.weight, dim=1),
        )
        scale = (self.parametrizations.weight.original / self.weight).nanmean()
        self._log_scale = nn.Parameter(softplus_inv(scale, beta=1.0e5))

    def forward(self, input: Tensor):
        scale = F.softplus(self._log_scale, beta=1.0e5)
        output = F.conv_transpose2d(
            input=input,
            weight=self.weight * scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )
        return output, scale


class SNSequential(nn.Sequential):
    def forward(self, input: Tensor):
        scales = []
        for layer in self:
            if isinstance(layer, (SNLinear, SNConv2d, SNConvTranspose2d)):
                input, scale = layer(input)
                scales.append(scale)
            else:
                input = layer(input)

        scale = torch.stack(scales).prod()
        return input, scale


class SNAtariDecoder(SNSequential):
    def __init__(
        self,
        in_features: int,
        hidden=48,
        out_channels=1,
    ):
        layers = [
            SNLinear(in_features, 32 * hidden),
            nn.ELU(),
            Reshape(32 * hidden, (32 * hidden, 1, 1)),
            SNConvTranspose2d(32 * hidden, 4 * hidden, 5, 2),
            nn.ELU(),
            SNConvTranspose2d(4 * hidden, 2 * hidden, 5, 2),
            nn.ELU(),
            SNConvTranspose2d(2 * hidden, hidden, 6, 2),
            nn.ELU(),
            SNConvTranspose2d(hidden, out_channels, 6, 2),
        ]

        super().__init__(*layers)
