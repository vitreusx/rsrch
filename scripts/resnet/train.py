import argparse
from pathlib import Path
from typing import ClassVar, Literal, TypedDict

import albumentations as A
import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import YAML
from torch import Tensor
from torch.profiler import ProfilerAction, ProfilerActivity
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.classification import Accuracy

from rsrch.data.imagenet import ImageNet
from rsrch.data.mnist import MNIST
from rsrch.exp import Experiment, boards
from rsrch.models import resnet
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.utils import cron, repro
from rsrch.utils.cast import cast
from rsrch.utils.ddp import auto_detect
from rsrch.utils.vis import make_grid

# isort: off
from config import Config, TimeDelta
# isort: on


class Item(TypedDict):
    """A dataset item for image classification."""

    image: Tensor  # (C, H, W), dtype: float
    label: int


class Batch(TypedDict):
    """A batch of items for image classification."""

    image: Tensor  # (N, C, H, W), dtype: float
    label: Tensor  # (N), dtype: long


class Dataset:
    """An adapter for `ImageNet` dataset for use in training ResNet."""

    MEAN: ClassVar = [0.485, 0.456, 0.406]  # "Canonical" ImageNet mean
    STD: ClassVar = [0.229, 0.224, 0.225]  # "Canonical" ImageNet std

    def __init__(
        self,
        base: ImageNet,
        transforms: list[A.ImageOnlyTransform],
        subset: list[int] | None = None,
    ):
        self.base = base
        # Metadata (ignore index, # of classes etc.) for the dataset
        self.meta = base.meta()

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

    def __getitem__(self, index: int) -> Item:
        item = self.base[self.indices[index]]
        image_nd = np.asarray(item["image"].convert("RGB"))
        item["image"] = self.img_transform(image=image_nd)["image"]
        return item

    def to_pil_image(self, image: torch.Tensor):
        image = image.detach()
        image = image.moveaxis(0, -1)  # [C, H, W] -> [H, W, C]
        mean = torch.tensor(self.MEAN, device=image.device)
        std = torch.tensor(self.STD, device=image.device)
        image = image * std + mean  # Invert the normalization transform
        image = (255 * image).clamp(0.0, 255.0).to(torch.uint8)
        image = Image.fromarray(image.cpu().numpy())
        return image

    @staticmethod
    def collate_fn(batch: list[Item]) -> Batch:
        image = torch.stack([item["image"] for item in batch])
        label = torch.tensor([item["label"] for item in batch])
        return {"image": image, "label": label}


class Trainer:
    project = "resnet"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        # Setup infra, data, models etc.
        self.setup()

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
                step_fn = lambda: getattr(self, delta.of)
                if mode == "every":
                    return cron.Every(step_fn, period=delta.n)
                elif mode == "until":
                    return cron.Until(step_fn, max_value=delta.n)
                else:
                    raise ValueError(mode)

        should_run = get_flag(self.cfg.train_for, mode="until")
        should_val = get_flag(self.cfg.val_every)
        should_save = get_flag(self.cfg.save_every)
        self.should_log = get_flag(self.cfg.log_every)
        self.should_save_samples = cron.OneTime()
        self.should_save_val_samples = cron.OneTime()

        # Training loop
        self.pbar = self.exp.make_pbar(desc="Train loop")
        while should_run:
            if should_val:
                self.val_epoch()
            if should_save:
                self.save_model(tag=f"model.step={self.step:07d}")
            self.train_step()
            self.step += 1
            self.pbar.update()

    def run_test(self):
        self.setup()

        self.should_log = True
        self.should_save_samples = True
        self.should_save_val_samples = True

        self.val_epoch()
        self.save_model(tag=f"model.step={self.step:07d}")
        self.train_step()

    def setup(self):
        self.setup_infra()
        self.setup_data()
        self.setup_loaders()
        self.setup_model()
        self.setup_prof()

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
        if self.cfg.dataset == "imagenet-100":
            self._setup_imagenet100()
        elif self.cfg.dataset == "mnist":
            self._setup_mnist()

        self.meta = self.train_data.meta

    def _setup_imagenet100(self):
        data_root = "./datasets/imagenet-100"
        self.in_channels = 3
        image_size = 224

        self.train_data = Dataset(
            ImageNet(data_root, split="train"),
            transforms=(
                A.Rotate(limit=(-30, 30), p=0.5),
                A.RandomResizedCrop((image_size, image_size)),
                A.HorizontalFlip(p=0.5),
            ),
        )

        val_ds = ImageNet(data_root, split="val")

        # For debugging, we limit the number of val samples
        if self.cfg.max_val_samples is not None:
            val_size = min(len(val_ds), self.cfg.max_val_samples)
            gen = np.random.default_rng()
            val_idxes = gen.choice(len(val_ds), size=val_size, replace=False)
            val_subset = val_idxes.tolist()
        else:
            val_subset = None

        self.val_data = Dataset(
            val_ds,
            transforms=(
                A.SmallestMaxSize(image_size),
                A.CenterCrop(image_size, image_size),
            ),
            subset=val_subset,
        )

    def _setup_mnist(self):
        data_root = "./datasets/mnist"
        self.in_channels = 1

        kw = {"mean": 0.5, "std": 0.5}

        train_ds = MNIST(data_root, split="train", download=True)
        self.train_data = Dataset(train_ds, **kw)

        val_ds = MNIST(data_root, split="test")

        # For debugging, we limit the number of val samples
        if self.cfg.max_val_samples is not None:
            val_size = min(len(val_ds), self.cfg.max_val_samples)
            gen = np.random.default_rng()
            val_idxes = gen.choice(len(val_ds), size=val_size, replace=False)
            val_subset = val_idxes.tolist()
        else:
            val_subset = None

        self.val_data = Dataset(val_ds, subset=val_subset, **kw)

    def setup_model(self):
        self.model: resnet.Resnet = getattr(resnet, self.cfg.model)(
            in_channels=self.in_channels,
            num_classes=self.meta.num_classes,
        )
        self.model = self.ddp.wrap_model(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        # Use `ScaledOptimizer` to simplify optimization step when autocasting
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)

    def setup_loaders(self):
        train_gen = torch.Generator()
        train_sampler = RandomSampler(self.train_data, generator=train_gen)
        train_sampler = self.ddp.wrap_sampler(
            sampler=train_sampler,
            set_epoch=lambda epoch: train_gen.manual_seed(epoch),
            drop_last=True,
        )

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            drop_last=True,
            collate_fn=self.train_data.collate_fn,
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
            drop_last=False,
            collate_fn=self.val_data.collate_fn,
        )

        self.train_iter = self.get_train_iter()

    def setup_prof(self):
        if not self.cfg.profile:
            return

        if self.ddp.num_replicas > 1:
            raise RuntimeError("Profiling is currently disabled for DDP.")

        activities = [ProfilerActivity.CPU]
        if self.ddp.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        def on_trace_ready(prof: torch.profiler.profile):
            dest = self.exp.dir / "trace.json.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(dest))
            self.exp.info("Saved trace data to %s", dest)

        self.is_profiling = False
        self.should_profile = cron.If(lambda: 128 <= self.step < 256)

        def schedule(step: int):  # noqa: ARG001
            if self.should_profile:
                if not self.is_profiling:
                    self.exp.info("Starting profiling")
                self.is_profiling = True
                return ProfilerAction.RECORD
            else:  # noqa: PLR5501
                if self.is_profiling:
                    self.exp.info("Stopping profiling")
                    self.is_profiling = False
                    return ProfilerAction.RECORD_AND_SAVE
                else:
                    return ProfilerAction.NONE

        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            with_stack=True,
            with_modules=True,
        )
        self.prof = self.prof.__enter__()

    def get_train_iter(self):
        self.epoch = 0
        while True:
            self.ddp.set_epoch(self.train_loader.sampler, self.epoch)
            yield from self.train_loader
            self.epoch += 1

    def train_step(self):
        batch = next(self.train_iter)
        batch: Batch = {k: v.to(self.ddp.device) for k, v in batch.items()}

        with self.autocast():
            logits: Tensor = self.model(batch["image"])
            loss = F.cross_entropy(logits, batch["label"])

        self.opt.step(loss)

        if self.cfg.profile and self.ddp.is_master:
            self.prof.step()

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
        gen = np.random.default_rng()
        idxes = gen.choice(num_images, size=num_samples, replace=False)

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
            batch: Batch = {k: v.to(self.ddp.device) for k, v in batch.items()}
            with self.autocast():
                logits: Tensor = self.model(batch["image"])
            top1.update(logits, batch["label"])
            if top5 is not None:
                top5.update(logits, batch["label"])

        top1_v = top1.compute()
        if top5 is not None:
            top5_v = top5.compute()

        if self.ddp.is_master:
            val_unit = self.cfg.val_every.of
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test",
        action="store_true",
        help="Run a regression test.",
    )
    args = p.parse_args()

    yaml = YAML(typ="safe", pure=True)
    with open(Path(__file__).parent / "config.yml", "r") as f:
        cfg = cast(yaml.load(f), Config)

    trainer = Trainer(cfg)
    if args.test:
        trainer.run_test()
    else:
        trainer.run()


if __name__ == "__main__":
    main()
