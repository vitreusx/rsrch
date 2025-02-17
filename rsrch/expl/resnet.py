from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor, nn
from torch.utils.data import DataLoader

from rsrch.data import IMAGENET_MEAN, IMAGENET_STD
from rsrch.data.imagenet import ImageNet
from rsrch.exp import Experiment, board
from rsrch.nn.optim import ScaledOptimizer
from rsrch.utils.data import Compose, InfiniteSampler, MapDict, Pipeline


class NormLayer(nn.Module):
    """A 'protocol class' for norm layers. Only used to provide type hints."""

    def __init__(self, num_features: int):
        ...


class ActLayer(nn.Module):
    """A 'protocol class' for activation layers. Only used to provide type hints."""

    def __init__(self, inplace: bool):
        ...


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: type[NormLayer] = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Residual path
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(in_channels)
        self.act = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(out_channels)

        # Skip path
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                norm_layer(out_channels),
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
        norm_layer: type[NormLayer] = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        # Residual path
        self.conv1 = nn.Conv2d(in_channels, bottleneck, 1, bias=False)
        self.bn1 = norm_layer(bottleneck)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, stride, 1, bias=False)
        self.bn2 = norm_layer(bottleneck)
        self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(bottleneck, out_channels, 1)
        self.bn3 = norm_layer(out_channels)

        # Skip path
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                norm_layer(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, input: Tensor):
        res = self.act1(self.bn1(self.conv1(input)))
        res = self.act2(self.bn2(self.conv2(res)))
        res = self.bn3(self.conv3(res))
        if self.downsample is not None:
            input = self.downsample(input)
        return input + res


class Resnet(nn.Module):
    def __init__(
        self,
        num_blocks: tuple[int, int, int, int],
        num_channels: tuple[int, int, int, int],
        block_type: Literal["basic", "bottleneck"] = "basic",
        num_classes: int = 1000,
        norm_layer: type[NormLayer] = nn.BatchNorm2d,
        act_layer: type[ActLayer] = nn.ReLU,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        if block_type == "bottleneck":
            bottleneck_sizes = [nc // 4 for nc in num_channels]
        else:
            bottleneck_sizes = [None] * 4

        self.layer1, self.layer2, self.layer3, self.layer4 = (
            self._make_layer(
                in_channels=64 if idx == 0 else num_channels[idx - 1],
                bottleneck=bottleneck_sizes[idx],
                out_channels=num_channels[idx],
                num_blocks=num_blocks[idx],
                downsample=(idx > 0),
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for idx in range(4)
        )

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
        input = self.relu(self.bn1(self.conv1(input)))
        input = self.layer1(self.maxpool(input))
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = self.avgpool(input).flatten(1)
        return self.fc(input)


def resnet18():
    return Resnet(
        num_blocks=[2, 2, 2, 2],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
    )


def resnet34():
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[64, 128, 256, 512],
        block_type="basic",
    )


def resnet50():
    return Resnet(
        num_blocks=[3, 4, 6, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
    )


def resnet101():
    return Resnet(
        num_blocks=[3, 4, 23, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
    )


def resnet152():
    return Resnet(
        num_blocks=[3, 8, 36, 3],
        num_channels=[256, 512, 1024, 2048],
        block_type="bottleneck",
    )


@dataclass
class Config:
    data_root: str = "datasets/imagenet-100"
    batch_size: int = 32
    device: str = "cuda"
    compute_dtype: str = "float16"


def main():
    cfg = Config()

    exp = Experiment(project="resnet")
    exp.add_board(board.WeightsAndBiases(dir=exp.dir, project=exp.project))

    step = 0
    exp.register_step("step", lambda: step, default=True)

    device = torch.device(cfg.device)
    compute_dtype = getattr(torch, cfg.compute_dtype)

    def autocast():
        return torch.autocast(
            device.type,
            compute_dtype,
            enabled=compute_dtype != torch.float32,
        )

    train_ds = Pipeline(
        ImageNet(
            root=cfg.data_root,
            split="train",
        ),
        MapDict(
            image=Compose(
                T.RandomResizedCrop(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ),
        ),
    )

    val_ds = Pipeline(
        ImageNet(
            root=cfg.data_root,
            split="val",
        ),
        MapDict(
            image=Compose(
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ),
        ),
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        sampler=InfiniteSampler(train_ds, shuffle=True),
        num_workers=2,
        pin_memory=device.type == "cuda",
        prefetch_factor=2,
    )
    train_iter = iter(train_loader)

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        prefetch_factor=2,
    )

    model = resnet18()
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt = ScaledOptimizer(opt)

    pbar = exp.make_pbar(desc="Resnet")
    while True:
        batch = next(train_iter)
        batch = {
            "image": batch["image"].to(device),
            "label": batch["label"].to(device),
        }

        with autocast():
            logits = model(batch["image"])
            loss = F.cross_entropy(logits, batch["label"])

        opt.step(loss)

        exp.add_scalar("train/loss", loss)

        step += 1
        pbar.update()


if __name__ == "__main__":
    main()
