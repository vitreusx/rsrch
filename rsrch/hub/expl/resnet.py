import torch
from torch import Tensor, nn


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Residual path
        # Why bias=False?
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        # Why bias=False?
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip path
        if in_channels != out_channels or stride > 1:
            # Why the BN here?
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, input: Tensor):
        res = self.act(self.bn1(self.conv1(input)))
        res = self.bn2(self.conv2(res))
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
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Residual path
        self.conv1 = nn.Conv2d(in_channels, bottleneck, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bottleneck, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip path
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Conv2d(in_channels, out_channels, stride, stride)
        else:
            self.downsample = None

    def forward(self, input: Tensor):
        res = self.act1(self.bn1(self.conv1(input)))
        res = self.act2(self.bn2(self.conv2(input)))
        res = self.bn3(self.conv3(input))
        if self.downsample is not None:
            input = self.downsample(input)
        return input + res


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()

        # Why is bias false and padding=3?
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Why is padding=1?
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(
            ResnetBlock(64, 64),
            ResnetBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResnetBlock(64, 128, stride=2),
            ResnetBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResnetBlock(128, 256, stride=2),
            ResnetBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            ResnetBlock(256, 512, stride=2),
            ResnetBlock(512, 512),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, input: Tensor):
        input = self.relu(self.bn1(self.conv1(input)))
        input = self.layer1(self.maxpool(input))
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.avgpool(input).flatten(1)
        return self.fc(input)


def main():
    ...


if __name__ == "__main__":
    main()
