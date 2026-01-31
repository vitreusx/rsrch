import math
import random
from typing import Literal, TypeAlias, TypedDict

import git
import numpy as np
import safetensors.torch as sft
import torch
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from rsrch.exp import Experiment
from rsrch.torch.nn.optim import ScaledOptimizer
from rsrch.utils import cron, repro
from rsrch.utils.ddp import DDPHelper
from rsrch.utils.vis import make_grid


class Sample(TypedDict):
    image: Tensor
    label: int


class Batch(TypedDict):
    images: Tensor
    labels: Tensor


class Dataset:
    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Sample:
        pass

    def collate_fn(self, batch: list[Sample]) -> Batch:
        pass

    def preview(self, sample: Sample) -> Image.Image:
        pass


TimeUnit: TypeAlias = Literal["steps", "samples", "epochs"]


class Config(BaseModel):
    project_name: str
    compute_dtype: str = "float32"
    fail_if_dirty: bool = True
    seed: int = 0
    device_batch_size: int
    total_batch_size: int | None = None
    num_dataloader_workers: int = 2
    max_train_epochs: int | None = None
    max_train_steps: int | None = None
    max_train_samples: int | None = None
    val_every: tuple[float, TimeUnit] | None = None
    save_model_every: tuple[float, TimeUnit] | None = None
    clip_grad: float | None = None
    sample_grid_size: tuple[int, int] = (4, 2)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.fail_if_dirty:
            repo = git.Repo()
            if repo.is_dirty():
                raise RuntimeError(
                    "Commit all changes in the repo, or disable `fail_if_dirty` option."
                )

    def run(self):
        self.setup_base()
        self.setup_data()
        self.setup_model()

    def setup_base(self):
        self.ddp = DDPHelper()

        repro.seed_all(self.cfg.seed)
        self.gen = np.random.default_rng(random.randint(0, 2**32 - 1))
        self.compute_dtype = getattr(torch, self.cfg.compute_dtype)

        if self.ddp.is_master:
            self.exp = Experiment(project=self.cfg.project_name)

        if self.cfg.total_batch_size is None:
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = math.ceil(
                self.cfg.total_batch_size
                / (self.ddp.world_size * self.cfg.device_batch_size)
            )

        self.total_batch_size = (
            self.gradient_accumulation_steps
            * self.ddp.world_size
            * self.cfg.device_batch_size
        )

    def setup_data(self):
        # Setup training data
        self.train_data = self.create_train_data()
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.device_batch_size,
            sampler=self.ddp.wrap_sampler(
                sampler=RandomSampler(self.train_data),
                drop_last=True,
            ),
            collate_fn=self.train_data.collate_fn,
            num_workers=self.cfg.num_dataloader_workers,
            worker_init_fn=repro.worker_init_fn(self.cfg.seed),
        )

        # Compute data stats - max number of opt steps, epochs and samples
        step_limits = []
        if self.cfg.max_train_epochs is not None:
            step_limits.append(
                int(
                    self.cfg.max_train_epochs
                    * len(self.train_loader)
                    / self.gradient_accumulation_steps
                )
            )
        if self.cfg.max_train_steps is not None:
            step_limits.append(self.cfg.max_train_steps)
        if self.cfg.max_train_samples is not None:
            step_limits.append(int(self.cfg.max_train_samples / self.total_batch_size))

        self.max_train_steps = min(step_limits)
        self.max_train_epochs = int(
            self.max_train_steps
            * self.gradient_accumulation_steps
            / len(self.train_loader)
        )
        self.max_train_samples = self.max_train_steps * self.total_batch_size

        # Setup validation data
        self.val_data = self.create_val_data()
        if self.val_data is not None:
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=self.cfg.device_batch_size,
                sampler=self.ddp.wrap_sampler(
                    sampler=SequentialSampler(self.val_data),
                    drop_last=False,
                ),
                collate_fn=self.val_data.collate_fn,
                num_workers=self.cfg.num_dataloader_workers,
            )

    def create_train_data(self) -> Dataset:
        raise NotImplementedError

    def create_val_data(self) -> Dataset | None:
        return None

    def setup_model(self):
        # Setup model
        self.model = self.create_model()
        self.model.to(self.ddp.device)
        self.model = self.ddp.wrap_model(self.model)

        # Setup optimizer
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = self.create_optimizer(self.parameters)
        self.opt = ScaledOptimizer(self.opt, self.compute_dtype)

    def create_model(self) -> nn.Module:
        raise NotImplementedError

    def create_optimizer(self, params: list[nn.Parameter]) -> Optimizer:
        raise NotImplementedError

    def run_train_loop(self):
        self.cur_epoch = self.cur_step = self.cur_sample = 0
        self.cur_iter = 0
        train_iter = self.get_train_iter()
        self.train_loss = 0.0

        # Add the steps to the dashboards
        self.exp.register_step("step", lambda: self.cur_step, default=True)
        self.exp.register_step("epoch", lambda: self.cur_epoch)
        self.exp.register_step("sample", lambda: self.cur_sample)

        # Set up flags for various actions to be done throughout training
        self.should_opt = self.make_flag((self.gradient_accumulation_steps, "iters"))
        self._val_step, should_val = self.make_flag(self.cfg.val_every)
        self._save_model_step, should_save_model = self.make_flag(
            self.cfg.save_model_every
        )

        # Run training loop
        self.show_data_samples()
        while self.cur_step < self.max_train_steps:
            if should_save_model:
                self.save_model()
            if should_val:
                self.val_epoch()
            batch = next(train_iter)
            self.train_step(batch)

    def get_train_iter(self):
        samples_per_iter = self.ddp.world_size * self.cfg.device_batch_size
        while True:
            self.ddp.set_epoch(self.train_loader.sampler, self.cur_epoch)
            for batch in self.train_loader:
                yield batch
                self.cur_iter += 1
                self.cur_sample += samples_per_iter
            self.cur_epoch += 1

    def make_flag(self, params: tuple[float, TimeUnit] | None = None):
        if params is None:
            return None, cron.Never()

        count, time_unit = params

        step_fn = {
            "steps": lambda: self.cur_step,
            "samples": lambda: self.cur_sample,
            "epochs": lambda: self.cur_epoch,
        }[time_unit]

        step_name = time_unit.removesuffix("s")
        flag = cron.Every(step_fn=step_fn, period=count)
        return step_name, flag

    def train_step(self, batch: Batch):
        with self.autocast():
            logits = self.model(batch["images"])
            loss = self.loss_fn(logits, batch["labels"])
            loss /= self.gradient_accumulation_steps
            loss.backward()

        self.train_loss += loss
        self.cur_iter += 1
        if self.cur_iter % self.gradient_accumulation_steps == 0:
            self.opt.step(self.train_loss, self.cfg.clip_grad)
            avg_loss = self.ddp.all_reduce(self.train_loss, "avg")
            if self.ddp.is_master:
                self.exp.add_scalar("train/loss", avg_loss, step="step")
            self.train_loss = 0.0
            self.cur_step += 1

    def autocast(self):
        return torch.autocast(self.ddp.device.type, self.compute_dtype)

    def loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError

    def show_data_samples(self):
        if self.ddp.is_master:
            for split, dataset in (("train", self.train_data), ("val", self.val_data)):
                gen = np.random.default_rng(random.randint(0, 2**32 - 1))
                rows, cols = self.cfg.sample_grid_size
                sample_indices = gen.choice(len(dataset), size=rows * cols).tolist()
                images = [dataset.preview(dataset[idx]) for idx in sample_indices]
                self.exp.add_image(
                    f"{split}/samples",
                    make_grid(images, ncols=cols),
                    step="step",
                )

    def val_epoch(self):
        self.model.eval()

        val_loss, val_acc = 0.0, 0.0
        for batch in self.val_loader:
            with self.autocast():
                logits = self.model(batch["images"])
                loss = self.loss_fn(logits, batch["labels"])
                preds = logits.argmax(-1)
            val_loss += loss.float()
            val_acc += (preds == batch["labels"]).float().mean()

        val_loss /= len(self.val_loader)
        val_loss = self.ddp.all_reduce(val_loss, "avg")

        val_acc /= len(self.val_loader)
        val_acc = self.ddp.all_reduce(val_acc, "avg")

        if self.ddp.is_master:
            self.exp.add_scalar("val/loss", val_loss, step=self._val_step)
            self.exp.add_scalar("val/acc", val_acc, step=self._val_step)

        self.model.train()

    def format_now(self, time_unit: TimeUnit):
        now, max_val = {
            "epochs": (self.cur_epoch, self.max_train_epochs),
            "steps": (self.cur_step, self.max_train_steps),
            "samples": (self.cur_sample, self.max_train_samples),
        }
        num_digits = math.ceil(math.log10(max_val + 1))
        now = f"%0{num_digits}d".format(now)
        return f"{time_unit}={now}"

    def save_model(self):
        if self.ddp.is_master:
            now = self.format_now(self._save_model_step)
            ckpt_path = self.exp.dir / "ckpts" / f"model.{now}.safetensors"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            sft.save_model(self.ddp.unwrap(self.model), str(ckpt_path))
