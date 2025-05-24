"""Feature extractors for various standard model architectures, like ResNet and VGG."""

from torch import Tensor, nn

from rsrch.models import resnet, vgg


class FromResnet(nn.Module):
    def __init__(self, model: resnet.Resnet):
        super().__init__()
        self.model = model

    def forward(self, input: Tensor) -> list[Tensor]:
        features = []

        feat2 = self.model.conv1(input)
        features.append(feat2)

        if self.model.bn1 is not None:
            x = self.model.bn1(feat2)
        x = self.model.maxpool(self.model.relu(x))

        for layer in self.model.layers:
            x = layer(x)
            if features[-1].shape[-2:] != x.shape[-2:]:
                # The block reduced input size, and so becomes a new feature map.
                features.append(x)
            else:
                # The block preserved input size. Given it's deeper, we replace
                # the previous feature map with it.
                features[-1] = x

        return features


class FromVGG(nn.Module):
    def __init__(self, model: vgg.VGG):
        super().__init__()
        self.blocks = nn.ModuleList(*model.blocks)

    def forward(self, input: Tensor) -> list[Tensor]:
        feats = []
        for block in self.blocks:
            input = block(input)
            feats.append(input)
        return feats
