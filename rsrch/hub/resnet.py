from typing import TYPE_CHECKING, Literal

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rsrch.data.imagenet import ImageNet
from rsrch.exp.boards.mlflow import MLflow
from rsrch.exp.boards.tensorboard import Tensorboard
from rsrch.hub import classifier
from rsrch.hub.classifier import Batch, Sample
from rsrch.models.resnet import resnet50

if TYPE_CHECKING:
    from rsrch.data.meta import ClsMeta


class Config(classifier.Config):
    board_type: Literal["tensorboard", "mlflow"] = "mlflow"
    dataset_type: Literal["imagenet"] = "imagenet"
    data_root: str
    max_val_samples: int | None
    lr: float = 3e-4


class TrainDataset(classifier.Dataset):
    """An adapter for `ImageNet` dataset for use in training ResNet."""

    MEAN = (0.485, 0.456, 0.406)  # "Canonical" ImageNet mean
    STD = (0.229, 0.224, 0.225)  # "Canonical" ImageNet std

    def __init__(
        self,
        base: Dataset,
        transforms: list[A.ImageOnlyTransform],
        subset: list[int] | None = None,
    ):
        self.base = base
        self.meta: ClsMeta = base.meta()

        if subset is None:
            self.indices = range(len(self.base))
        else:
            self.indices = subset

        self.img_transform = A.Compose(
            [
                *transforms,
                A.Normalize(self.MEAN, self.STD),
                A.ToTensorV2(),
            ],
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        item = self.base[self.indices[index]]
        image_nd = np.asarray(item["image"].convert("RGB"))
        item["image"] = self.img_transform(image=image_nd)["image"]
        return item

    def preview(self, image: torch.Tensor):
        image = image.detach()
        image = image.moveaxis(0, -1)  # [C, H, W] -> [H, W, C]
        mean = torch.tensor(self.MEAN, device=image.device)
        std = torch.tensor(self.STD, device=image.device)
        image = image * std + mean  # Invert the normalization transform
        image = (255 * image).clamp(0.0, 255.0).to(torch.uint8)
        image = Image.fromarray(image.cpu().numpy())
        return image

    @staticmethod
    def collate_fn(batch: list[Sample]) -> Batch:
        images = torch.stack([item["image"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return {"images": images, "labels": labels}


class Trainer(classifier.Trainer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg: Config

    def setup_base(self):
        super().setup_base()

        if self.cfg.board_type == "tensorboard":
            board = Tensorboard(dir=self.exp.dir, launch=True)
        elif self.cfg.board_type == "mlflow":
            board = MLflow(exp_name=self.exp.project)
        self.exp.add_board(board)

    def create_train_data(self):
        if self.cfg.dataset_type == "imagenet":
            self._in_channels = 3
            return TrainDataset(
                ImageNet(root=self.cfg.data_root, split="train"),
                transforms=[
                    A.Rotate(limit=(-30, 30), p=0.5),
                    A.RandomResizedCrop((224, 224)),
                    A.HorizontalFlip(p=0.5),
                ],
            )

        msg = f"Unknown dataset type {self.cfg.dataset_type}"
        raise ValueError(msg)

    def create_val_data(self):
        if self.cfg.dataset_type == "imagenet":
            val_data = ImageNet(root=self.cfg.data_root, split="val")
            transforms = [A.CenterCrop(224, 224)]

        if self.cfg.max_val_samples is None:
            val_subset = None
        elif self.cfg.max_val_samples <= 0:
            return None
        else:
            val_samples = min(len(val_data), self.cfg.max_val_samples)
            val_subset = self.gen.choice(len(val_data), size=val_samples)

        return TrainDataset(
            val_data,
            transforms=transforms,
            subset=val_subset,
        )

    def create_model(self):
        return resnet50(
            in_channels=self._in_channels,
            num_classes=self.train_data.meta.num_classes,
        )

    def create_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.cfg.lr)
