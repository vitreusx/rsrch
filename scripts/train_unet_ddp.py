from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict

import albumentations as A
import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import MeanIoU

from rsrch.data.voc import VOCSegmentation
from rsrch.exp import Experiment, boards
from rsrch.models.unet import UNet
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.utils import cron, repro
from rsrch.utils.ddp import auto_detect
from rsrch.utils.download import download_url
from rsrch.utils.preview import make_grid


class TimeDelta(TypedDict):
    n: int
    of: Literal["step", "epoch"]


class Config:
    seed: int = 0
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    batch_size: int = 16
    val_batch_size: int | None = None
    log_every: TimeDelta = {"n": 4, "of": "step"}
    val_every: TimeDelta = {"n": 256, "of": "step"}
    save_every: TimeDelta | None = None
    resize_mode: Literal["preds", "labels"] = "preds"


class Item(TypedDict):
    image: Tensor
    labels: Tensor


class Batch(TypedDict):
    image: Tensor
    labels: Tensor


def collate_fn(batch: list[Item]) -> Batch:
    image = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"image": image, "labels": labels}


def move_to_device(batch: Batch, device: torch.device) -> Batch:
    return {
        "image": batch["image"].to(device),
        "labels": batch["labels"].to(device),
    }


def resize(input: Tensor, other: Tensor):
    if input.dtype.is_floating_point:
        interp_mode = tv_F.InterpolationMode.BILINEAR
    else:
        interp_mode = tv_F.InterpolationMode.NEAREST
    return tv_F.resize(input, other.shape[-2:], interp_mode)


class Trainer:
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
                step_fn = lambda: getattr(self, delta["of"])
                return cron.Every(step_fn=step_fn, period=delta["n"])

        should_val = get_flag(self.cfg.val_every)
        should_save = get_flag(self.cfg.save_every)
        self.should_log = get_flag(self.cfg.log_every)
        self.should_save_samples = cron.Once()
        self.should_save_val_samples = cron.Always()

        while True:
            if should_val:
                self.val_epoch()
            if should_save:
                tag = f"model.step={self.step:07d}"
                self.save_model(tag)
            self.train_step()
            self.step += 1

    def setup_infra(self):
        self.ddp = auto_detect()
        repro.seed_all(self.cfg.seed)
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        self.step, self.epoch = 0, 0
        if self.ddp.is_master:
            self.exp = Experiment(project="unet")
            self.exp.add_board(boards.Tensorboard(self.exp.dir / "board", launch=True))

            self.exp.register_step("step", lambda: self.step)
            self.exp.register_step("epoch", lambda: self.epoch)

    def setup_data(self):
        voc_root = "./datasets/voc"
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])

        class Data:
            def __init__(
                self,
                split: Literal["train", "val"],
                transforms: list,
            ):
                super().__init__()
                self.base = VOCSegmentation(voc_root, split=split)

                self.meta = self.base.meta()

                self.transform = A.Compose(
                    [
                        *transforms,
                        A.Normalize(MEAN, STD),
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
                img_nd = img_nd * STD + MEAN
                img_nd = (255 * img_nd).astype(np.uint8)
                return Image.fromarray(img_nd)

        img_size = 256

        self.train_data = Data(
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

        self.val_data = Data(
            split="val",
            transforms=[
                A.SmallestMaxSize(img_size),
                A.CenterCrop(img_size, img_size),
            ],
        )

        self.meta = self.train_data.meta

    def setup_model(self):
        self.model = UNet(
            in_channels=3,
            out_channels=self.meta.num_classes,
            block_channels=[32, 64, 128, 256],
        )
        self.model = self.ddp.wrap_model(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)

    def setup_data_loaders(self):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            sampler=self.ddp.get_sampler(
                self.train_data,
                shuffle=True,
                seed=0,
                drop_last=True,
            ),
            num_workers=2,
            worker_init_fn=repro.worker_init_fn,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.cfg.val_batch_size or self.cfg.batch_size,
            sampler=self.ddp.get_sampler(
                self.val_data,
                shuffle=False,
                seed=0,
                drop_last=False,
            ),
            num_workers=2,
            worker_init_fn=repro.worker_init_fn,
            collate_fn=collate_fn,
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
            logits: Tensor = self.model(batch["image"])

            if self.cfg.resize_mode == "labels":
                labels = resize(batch["labels"], logits)
            else:
                labels = batch["labels"]
                logits = resize(logits, labels)

            if self.train_data.meta.ignore_index == 0:
                labels = labels - 1

            loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=self.meta.ignore_index,
            )

        self.opt.step(loss)

        if self.should_log:
            self.ddp.all_reduce(loss, op="mean")
            if self.ddp.is_master:
                self.exp.add_scalar("train/loss", loss, step="step")

    def get_sample_grid(
        self,
        dataset,
        batch: Batch,
        logits: Tensor,
    ):
        num_images = len(batch["image"])
        num_samples = min(num_images, 8)
        idxes = np.random.choice(num_images, size=num_samples, replace=False)

        to_pil_image = dataset.to_pil_image
        palette = self.meta.palette
        ignore_index = self.meta.ignore_index

        if self.cfg.resize_mode == "labels":
            labels = resize(batch["labels"], logits)
        else:
            labels = batch["labels"]
            logits = resize(logits, labels)

        grid = []
        for idx in idxes:
            img = to_pil_image(batch["image"][idx].cpu())

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
            ignore_index=self.meta.ignore_index,
        ).to(self.ddp.device)

        for batch in self.val_loader:
            batch = move_to_device(batch, self.ddp.device)
            with self.autocast():
                logits: Tensor = self.model(batch["image"])
            preds = logits.argmax(1)
            preds = resize(preds, batch["labels"])
            mean_iou.update(preds, batch["labels"])

        if self.ddp.is_master:
            val_unit = self.cfg.val_every["of"]
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
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
