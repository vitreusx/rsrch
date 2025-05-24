from typing import Literal

from torch import Tensor, nn

from rsrch.utils.state_dict import leaves, to_tree

from .types import ActLayer, NormLayer


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        if out_channels is None:
            out_channels = in_channels

        # Residual path
        if norm_layer is None:
            bn1 = bn2 = None
            bias = True
        else:
            bn1 = norm_layer(out_channels)
            bn2 = norm_layer(out_channels)
            bias = False

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=bias)
        self.bn1 = bn1
        self.act = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=bias)
        self.bn2 = bn2

        # Skip path
        if in_channels != out_channels or stride > 1:
            conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=bias)
            if norm_layer is None:
                self.downsample = conv
            else:
                norm = norm_layer(out_channels)
                self.downsample = nn.Sequential(conv, norm)
        else:
            self.downsample = None

    def forward(self, input: Tensor):
        res = self.conv1(input)
        if self.bn1 is not None:
            res = self.bn1(res)
        res = self.act(self.conv2(res))
        if self.bn2 is not None:
            res = self.bn2(res)
        if self.downsample is not None:
            input = self.downsample(input)
        return input + res


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck: int,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck = bottleneck
        self.out_channels = out_channels
        self.stride = stride
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        if out_channels is None:
            out_channels = in_channels

        # Residual path
        if norm_layer is None:
            bn1 = bn2 = bn3 = None
            bias = True
        else:
            bn1 = norm_layer(bottleneck)
            bn2 = norm_layer(bottleneck)
            bn3 = norm_layer(out_channels)
            bias = False

        # Original ResNet adds stride in conv1, but torchvision adds it on
        # conv2. See
        #   https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        # for the rationale

        self.conv1 = nn.Conv2d(in_channels, bottleneck, 1, bias=bias)
        self.bn1 = bn1
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, stride, 1, bias=bias)
        self.bn2 = bn2
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(bottleneck, out_channels, 1, bias=bias)
        self.bn3 = bn3

        # Skip path
        if in_channels != out_channels or stride > 1:
            conv = nn.Conv2d(in_channels, out_channels, 1, stride, bias=bias)
            if norm_layer is None:
                self.downsample = conv
            else:
                self.downsample = nn.Sequential(conv, norm_layer(out_channels))
        else:
            self.downsample = None

    def forward(self, input: Tensor):
        res = self.conv1(input)
        if self.bn1 is not None:
            res = self.bn1(res)
        res = self.conv2(self.act1(res))
        if self.bn2 is not None:
            res = self.bn2(res)
        res = self.conv3(self.act2(res))
        if self.bn3 is not None:
            res = self.bn3(res)
        if self.downsample is not None:
            input = self.downsample(input)
        return input + res


class Resnet(nn.Module):
    def __init__(
        self,
        num_blocks: list[int],
        num_channels: list[int],
        block_type: Literal["basic", "bottleneck"] = "basic",
        in_channels: int = 3,
        num_classes: int | None = 1000,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.block_type = block_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.act_layer = act_layer

        assert len(num_blocks) == len(num_channels)

        if norm_layer is None:
            bn1 = None
            bias = True
        else:
            bn1 = norm_layer(64)
            bias = False

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=bias)
        self.bn1 = bn1
        self.relu = act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        if block_type == "bottleneck":
            bottleneck_sizes = [nc // 4 for nc in num_channels]
        else:
            bottleneck_sizes = [None] * len(num_blocks)

        self.layers = nn.Sequential(
            *(
                self._make_layer(
                    in_channels=64 if idx == 0 else num_channels[idx - 1],
                    bottleneck=bottleneck_sizes[idx],
                    out_channels=num_channels[idx],
                    num_blocks=num_blocks[idx],
                    downsample=(idx > 0),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for idx in range(len(num_blocks))
            )
        )

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(num_channels[3], num_classes)

    def _make_layer(
        self,
        in_channels: int,
        bottleneck: int | None,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
        norm_layer: type[NormLayer],
        act_layer: type[ActLayer],
    ):
        layers = []
        for block_idx in range(num_blocks):
            stride = 2 if downsample and block_idx == 0 else 1
            if bottleneck is None:
                layer = BasicBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            else:
                layer = Bottleneck(
                    in_channels=in_channels,
                    bottleneck=bottleneck,
                    out_channels=out_channels,
                    stride=stride,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            layers.append(layer)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, input: Tensor):
        input = self.conv1(input)
        if self.bn1 is not None:
            input = self.bn1(input)
        input = self.maxpool(self.relu(input))

        input = self.layers(input)

        if self.num_classes is not None:
            input = self.avgpool(input)
            output = self.fc(input.flatten(1))
        else:
            output = input

        return output


def _resnet_factory(func):
    def wrapped(
        in_channels: int = 3,
        num_classes: int = 1000,
        norm_layer: type[NormLayer] | None = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ) -> Resnet:
        return func(
            in_channels=in_channels,
            num_classes=num_classes,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    return wrapped


@_resnet_factory
def resnet18(**kwargs):
    return Resnet(
        num_blocks=[2, 2, 2, 2],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
        **kwargs,
    )


@_resnet_factory
def resnet34(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
        **kwargs,
    )


@_resnet_factory
def resnet50(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


@_resnet_factory
def resnet101(**kwargs):
    return Resnet(
        num_blocks=[3, 4, 23, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


@_resnet_factory
def resnet152(**kwargs):
    return Resnet(
        num_blocks=[3, 8, 36, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
        **kwargs,
    )


def adapt_tv_state_dict(state_dict: dict[str, Tensor]):
    # Convert flat state dict to a tree
    state_dict = to_tree(state_dict)

    # Rename layer0 to layers.0 etc.
    rename = {f"layer{idx+1}": f"layers.{idx}" for idx in range(4)}
    for k in [*state_dict]:
        if k in rename:
            v = state_dict[k]
            del state_dict[k]
            state_dict[rename[k]] = v

    # Convert tree state dict back to flat repr
    return leaves(state_dict)
