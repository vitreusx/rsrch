from torch import Tensor, nn

from .types import ActLayer


class Block(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        final_conv_1x1: bool = False,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        layers = []
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1 and final_conv_1x1:
                kernel_size = 1
                padding = 0
            else:
                kernel_size = 3
                padding = 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding))
            layers.append(act_layer())
            in_channels = out_channels
        layers.append(nn.MaxPool2d((2, 2)))

        super().__init__(*layers)


class VGG(nn.Module):
    def __init__(
        self,
        block_channels: list[int],
        block_num_layers: list[int],
        in_channels: int = 3,
        num_classes: int = 1000,
        fc_features: int = 4096,
        num_fc_layers: int = 2,
        final_conv_1x1: bool | list[bool] = False,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()

        self.num_blocks = len(block_channels)
        if isinstance(final_conv_1x1, bool):
            final_conv_1x1 = [final_conv_1x1] * self.num_blocks

        self.blocks = nn.Sequential()
        for idx in range(self.num_blocks):
            self.blocks.append(
                Block(
                    in_channels=in_channels if idx == 0 else block_channels[idx - 1],
                    out_channels=block_channels[idx],
                    num_layers=block_num_layers[idx],
                    final_conv_1x1=final_conv_1x1[idx],
                    act_layer=act_layer,
                )
            )

        self.fc = nn.Sequential()
        for idx in range(num_fc_layers):
            in_features = block_channels[-1] * 7 * 7 if idx == 0 else fc_features
            out_features = fc_features
            self.fc.append(nn.Linear(in_features, out_features))
            self.fc.append(act_layer())
        self.fc.append(nn.Linear(fc_features, num_classes))

    def forward(self, input: Tensor) -> Tensor:
        features: Tensor = self.blocks(input)
        features = features.flatten(1)
        return self.fc(features)


def _vgg_factory(func):
    def wrapped(
        in_channels: int = 3,
        num_classes: int = 1000,
        fc_features: int = 4096,
        num_fc_layers: int = 2,
        final_conv_1x1: bool | list[bool] = False,
        act_layer: type[ActLayer] = nn.ReLU,
    ) -> VGG:
        return func(
            in_channels=in_channels,
            num_classes=num_classes,
            fc_features=fc_features,
            num_fc_layers=num_fc_layers,
            final_conv_1x1=final_conv_1x1,
            act_layer=act_layer,
        )

    return wrapped


@_vgg_factory
def vgg11(**kwargs):
    """VGG11 network. Corresponds to Config A from the paper."""
    return VGG(
        block_channels=[64, 128, 256, 512],
        block_num_layers=[1, 1, 2, 2, 2],
        **kwargs,
    )


@_vgg_factory
def vgg13(**kwargs):
    """VGG13 network. Corresponds to Config C from the paper."""
    return VGG(
        block_channels=[64, 128, 256, 512],
        block_num_layers=[2, 2, 2, 2, 2],
        **kwargs,
    )


@_vgg_factory
def vgg16(**kwargs):
    """VGG16 network. Corresponds to Config D from the paper."""
    return VGG(
        block_channels=[64, 128, 256, 512],
        block_num_layers=[2, 2, 3, 3, 3],
        **kwargs,
    )


@_vgg_factory
def vgg19(**kwargs):
    """VGG19 network. Corresponds to Config E from the paper."""
    return VGG(
        block_channels=[64, 128, 256, 512],
        block_num_layers=[2, 2, 4, 4, 4],
        **kwargs,
    )
