from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict

import albumentations as A
import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from rsrch.data.imagenet import ImageNet
from rsrch.data.voc import VOCSegmentation
from rsrch.exp import Experiment, boards
from rsrch.models.resnet import convert_from_torchvision, resnet18, resnet34
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
    train_for: TimeDelta = {"n": int(100e3), "of": "step"}
    log_every: TimeDelta = {"n": 4, "of": "step"}
    val_every: TimeDelta = {"n": 256, "of": "step"}
    save_every: TimeDelta | None = None
    resize_mode: Literal["preds", "labels"] = "preds"


class Item(TypedDict):
    image: Tensor
    label: int


class Batch(TypedDict):
    image: Tensor
    label: Tensor


def collate_fn(batch: list[Item]) -> Batch:
    image = torch.stack([item["image"] for item in batch])
    label = torch.tensor([item["label"] for item in batch])
    return {"image": image, "label": label}


def move_to_device(batch: Batch, device: torch.device) -> Batch:
    return {
        "image": batch["image"].to(device),
        "label": batch["label"].to(device),
    }


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        # Setup infra, data, models etc.
        self.setup_infra()
        self.setup_data()
        self.setup_data_loaders()
        self.setup_model()

        # Setup loop control flags
        def get_flag(
            delta: TimeDelta | None,
            mode: Literal["every", "until"] = "every",
        ):
            if delta is None:
                if mode == "every":
                    return cron.Never()
                else:
                    return cron.Always()
            else:
                step_fn = lambda: getattr(self, delta["of"])
                if mode == "every":
                    return cron.Every(step_fn, period=delta["n"])
                elif mode == "until":
                    return cron.Until(step_fn, max_value=delta["n"])

        should_run = get_flag(self.cfg.train_for, mode="until")
        should_val = get_flag(self.cfg.val_every)
        should_save = get_flag(self.cfg.save_every)
        self.should_log = get_flag(self.cfg.log_every)
        self.should_save_samples = cron.Once()
        self.should_save_val_samples = cron.Once()

        # Training loop
        while should_run:
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
            self.exp = Experiment(project="resnet")
            self.exp.add_board(boards.Tensorboard(self.exp.dir / "board", launch=True))

            self.exp.register_step("step", lambda: self.step)
            self.exp.register_step("epoch", lambda: self.epoch)

    def setup_data(self):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        class Data:
            def __init__(
                self,
                base: ImageNet,
                transforms: list[A.ImageOnlyTransform],
                subset: list[int] | None = None,
            ):
                self.base = base
                self.meta = base.meta()

                if subset is None:
                    self.indices = range(len(self.base))
                else:
                    self.indices = subset

                self.img_transform = A.Compose(
                    [
                        *transforms,
                        A.Normalize(MEAN, STD),
                        A.ToTensorV2(),
                    ]
                )

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, index: int):
                item = self.base[self.indices[index]]
                image_nd = np.asarray(item["image"].convert("RGB"))
                item["image"] = self.img_transform(image=image_nd)["image"]
                return item

            def to_pil_image(self, image: torch.Tensor):
                image = image.detach()
                image = image.moveaxis(0, -1)  # [C, H, W] -> [H, W, C]
                mean = torch.tensor(MEAN, device=image.device)
                std = torch.tensor(STD, device=image.device)
                image = image * std + mean
                image = (255 * image).clamp(0.0, 255.0).to(torch.uint8)
                image = Image.fromarray(image.cpu().numpy())
                return image

        data_root = "./datasets/imagenette-x"

        self.train_data = Data(
            base=ImageNet(data_root, split="train"),
            transforms=(
                A.Rotate(limit=(-30, 30), p=0.5),
                A.RandomResizedCrop((224, 224)),
                A.HorizontalFlip(p=0.5),
            ),
        )

        val_ds = ImageNet(data_root, split="val")
        val_size = min(len(val_ds), 1024)
        val_idxes = np.random.choice(len(val_ds), size=val_size, replace=False)

        self.val_data = Data(
            base=val_ds,
            transforms=(
                A.SmallestMaxSize(224),
                A.CenterCrop(224, 224),
            ),
            subset=val_idxes.tolist(),
        )

        self.meta = self.train_data.meta

    def setup_model(self):
        self.model = resnet34(num_classes=self.meta.num_classes)
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
            loss = F.cross_entropy(logits, batch["label"])

        self.opt.step(loss)

        if self.should_log:
            self.ddp.all_reduce(loss, op="mean")
            if self.ddp.is_master:
                self.exp.add_scalar("train/loss", loss, step="step")
                if self.should_save_samples:
                    samples = self.get_sample_grid(self.train_data, batch)
                    self.exp.add_image("train/samples", samples, step="step")

    def get_sample_grid(self, dataset, batch: Batch):
        num_images = len(batch["image"])
        num_samples = min(num_images, 8)
        idxes = np.random.choice(num_images, size=num_samples, replace=False)

        to_pil_image = dataset.to_pil_image
        images = [to_pil_image(batch["image"][idx]) for idx in idxes]

        return make_grid(images, ncols=4)

    @torch.no_grad()
    def val_epoch(self):
        task = "binary" if self.meta.num_classes == 2 else "multiclass"
        top1 = Accuracy(task=task, top_k=1, num_classes=self.meta.num_classes)
        top1.to(self.ddp.device)

        top5 = None
        if self.meta.num_classes >= 5:
            top5 = Accuracy(task=task, top_k=5, num_classes=self.meta.num_classes)
            top5.to(self.ddp.device)

        for batch in self.val_loader:
            batch = move_to_device(batch, self.ddp.device)
            with self.autocast():
                logits: Tensor = self.model(batch["image"])
            top1.update(logits, batch["label"])
            if top5 is not None:
                top5.update(logits, batch["label"])

        top1_v = top1.compute()
        if top5 is not None:
            top5_v = top5.compute()

        if self.ddp.is_master:
            val_unit = self.cfg.val_every["of"]
            self.exp.add_scalar("val/acc", top1_v, step=val_unit)
            if top5 is not None:
                self.exp.add_scalar("val/acc_top5", top5_v, step=val_unit)
            if self.should_save_val_samples:
                samples = self.get_sample_grid(self.val_data, batch)
                self.exp.add_image("val/samples", samples, step="step")

    def save_model(self, tag: str):
        if self.ddp.is_master:
            state = self.ddp.state_dict(self.model)
            dest = self.exp.dir / "ckpts" / f"{tag}.safetensors"
            dest.parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(state, dest)

    def autocast(self):
        return torch.autocast(
            device_type=self.ddp.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype != torch.float32,
        )


def main():
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
