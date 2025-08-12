from pathlib import Path
from typing import Literal, TypedDict

import albumentations as A
import numpy as np
import safetensors
import safetensors.torch

# from rsrch.models.unet import UNet
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image
from ruamel.yaml import YAML
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics import JaccardIndex

from rsrch.data.voc import VOCSegmentation
from rsrch.exp import Experiment, boards
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.utils import cron, repro
from rsrch.utils.cast import cast
from rsrch.utils.ddp import auto_detect
from rsrch.utils.preview import make_grid

# isort: off
from config import Config, TimeDelta
# isort: on


class Item(TypedDict):
    image: Tensor
    labels: Tensor


class Batch(TypedDict):
    image: Tensor
    labels: Tensor


class Dataset:
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        transforms: list,
    ):
        super().__init__()
        self.base = VOCSegmentation(root, split=split)

        self.meta = self.base.meta()

        self.transform = A.Compose(
            [
                *transforms,
                A.Normalize(self.MEAN, self.STD),
                A.ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        image = np.asarray(item["image"].convert("RGB"))
        labels = np.asarray(item["labels"])
        res = self.transform(image=image, mask=labels)
        return {
            "image": res["image"],
            "labels": res["mask"].to(torch.long),
        }

    def to_pil_image(self, image: Tensor):
        img_nd = image.moveaxis(0, -1).numpy(force=True)
        img_nd = img_nd * self.STD + self.MEAN
        img_nd = (255 * img_nd).astype(np.uint8)
        return Image.fromarray(img_nd)

    @staticmethod
    def collate_fn(batch: list[Item]) -> Batch:
        image = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"image": image, "labels": labels}


def move_to_device(batch: Batch, device: torch.device) -> Batch:
    return {
        "image": batch["image"].to(device),
        "labels": batch["labels"].to(device),
    }


def match_size(logits: Tensor, labels: Tensor):
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = tv_F.resize(logits, labels.shape[-2:])
    return logits


class Trainer:
    project = "unet"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        self.setup_infra()
        self.setup_data()
        self.setup_data_loaders()
        self.setup_model()

        def get_flag(delta: TimeDelta | None):
            if delta is None:
                return cron.Never()
            else:
                step_fn = lambda: getattr(self, delta.of)
                return cron.Every(step_fn=step_fn, period=delta.n)

        should_val = get_flag(self.cfg.val_every)
        should_save = get_flag(self.cfg.save_every)
        self.should_log = get_flag(self.cfg.log_every)
        self.should_save_samples = cron.Once()
        self.should_save_val_samples = cron.Always()

        self.pbar = self.exp.make_pbar(desc=self.project)
        while True:
            if should_val:
                self.val_epoch()
            if should_save:
                tag = f"model.step={self.step:07d}"
                self.save_model(tag)
            self.train_step()
            self.step += 1
            self.pbar.update()

    def setup_infra(self):
        self.ddp = auto_detect()
        repro.seed_all(self.cfg.seed)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        self.step, self.epoch = 0, 0
        if self.ddp.is_master:
            self.exp = Experiment(
                project=self.project,
                create_commit=self.cfg.create_exp_commit,
            )
            self.exp.add_board(boards.Tensorboard(self.exp.dir / "board", launch=True))

            self.exp.register_step("step", lambda: self.step)
            self.exp.register_step("epoch", lambda: self.epoch)

    def setup_data(self):
        voc_root = "datasets/voc"
        img_size = 256

        self.train_data = Dataset(
            root=voc_root,
            split="train",
            transforms=[
                A.RandomResizedCrop(
                    (img_size, img_size),
                    scale=(0.08, 1.0),
                    ratio=(3 / 4, 4 / 3),
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.5),
            ],
        )

        self.val_data = Dataset(
            root=voc_root,
            split="val",
            transforms=[
                A.SmallestMaxSize(img_size),
                A.CenterCrop(img_size, img_size),
            ],
        )

        self.meta = self.train_data.meta
        if self.train_data.meta.ignore_index == 0:
            self.reduce_zero = True
            self.ignore_index = -1
        else:
            self.reduce_zero = False
            self.ignore_index = self.train_data.meta.ignore_index

    def setup_model(self):
        # h = 32
        # self.model = UNet(
        #     in_channels=3,
        #     out_channels=self.meta.num_classes,
        #     block_channels=[h, 2 * h, 4 * h, 8 * h],
        # )
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            decoder_interpolation="bilinear",
            in_channels=3,
            classes=self.meta.num_classes,
        )
        self.model = self.ddp.wrap_model(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)

    def setup_data_loaders(self):
        train_gen = torch.Generator()
        train_sampler = self.ddp.wrap_sampler(
            RandomSampler(self.train_data, generator=train_gen),
            set_epoch=lambda epoch: train_gen.manual_seed(epoch),
            drop_last=True,
        )

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            num_workers=2,
            worker_init_fn=repro.worker_init_fn,
            collate_fn=self.train_data.collate_fn,
            persistent_workers=True,
        )

        val_sampler = range(len(self.val_data))
        val_sampler = self.ddp.wrap_sampler(
            val_sampler,
            set_epoch=None,
            drop_last=False,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.cfg.val_batch_size or self.cfg.batch_size,
            sampler=val_sampler,
            num_workers=2,
            worker_init_fn=repro.worker_init_fn,
            collate_fn=self.val_data.collate_fn,
            persistent_workers=True,
        )

        self.train_iter = self.get_train_iter()

    def get_train_iter(self):
        self.epoch = 0
        while True:
            self.ddp.set_epoch(self.train_loader.sampler, self.epoch)
            yield from self.train_loader
            self.epoch += 1

    def train_step(self):
        batch = next(self.train_iter)
        batch = move_to_device(batch, self.ddp.device)

        with self.autocast():
            labels = batch["labels"]
            if self.reduce_zero:
                labels = labels - 1

            logits: Tensor = self.model(batch["image"])
            logits = match_size(logits, labels)

            loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=self.ignore_index,
            )

        self.opt.step(loss)

        if self.should_log:
            self.ddp.all_reduce(loss, op="mean")
            if self.ddp.is_master:
                self.exp.add_scalar("train/loss", loss, step="step")

    def get_sample_grid(self, dataset: Dataset, batch: Batch, logits: Tensor):
        num_images = len(batch["image"])
        num_samples = min(num_images, 8)
        idxes = np.random.choice(num_images, size=num_samples, replace=False)

        palette = self.meta.palette
        ignore_index = self.meta.ignore_index

        labels = batch["labels"]
        logits = match_size(logits, labels)

        grid = []
        for idx in idxes:
            img = dataset.to_pil_image(batch["image"][idx].cpu())

            seg_map_gt = labels[idx].numpy(force=True)
            seg_map_gt = palette.label2rgb(seg_map_gt, ignore_index)
            seg_map_gt = Image.fromarray(seg_map_gt)

            seg_map_pred = logits[idx].argmax(0).numpy(force=True)
            seg_map_pred = palette.label2rgb(seg_map_pred)
            seg_map_pred = Image.fromarray(seg_map_pred)

            grid.append([img, seg_map_gt, seg_map_pred])

        return make_grid(grid)

    @torch.no_grad()
    def val_epoch(self):
        task = "binary" if self.meta.num_classes == 2 else "multiclass"
        mean_iou = JaccardIndex(
            task=task,
            num_classes=self.meta.num_classes,
            ignore_index=self.ignore_index,
        ).to(self.ddp.device)

        for batch in self.val_loader:
            batch = move_to_device(batch, self.ddp.device)

            with self.autocast():
                logits: Tensor = self.model(batch["image"])
            preds = logits.argmax(1)

            labels = batch["labels"]
            if self.reduce_zero:
                labels = labels - 1

            logits = match_size(logits, labels)
            mean_iou.update(preds, batch["labels"])

        if self.ddp.is_master:
            val_unit = self.cfg.val_every.of
            self.exp.add_scalar("val/mean_iou", mean_iou.compute(), step=val_unit)

            if self.should_save_val_samples:
                samples = self.get_sample_grid(self.val_data, batch, logits)
                self.exp.add_image("val/samples", samples, step=val_unit)

    def save_model(self, tag: str):
        if self.ddp.is_master:
            state = self.ddp.state_dict(self.model)
            dest = self.exp.dir / "ckpts" / f"{tag}.safetensors"
            dest.parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(state, dest)

    def autocast(self):
        return torch.autocast(
            self.ddp.device.type,
            self.compute_dtype,
            enabled=self.compute_dtype != torch.float32,
        )


def main():
    yaml = YAML(typ="safe", pure=True)
    with open(Path(__file__).parent / "config.yml", "r") as f:
        cfg = cast(yaml.load(f), Config)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
