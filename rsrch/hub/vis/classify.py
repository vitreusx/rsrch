from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

import torchmetrics
from rsrch.nn.optim import ScaledOptimizer
from rsrch.utils.config import Dynamic
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from rsrch.exp import Experiment
from rsrch.utils import cron
import torch.nn.functional as F
import safetensors.torch


@dataclass
class _Time:
    n: float
    of: str


Time = float | _Time


@dataclass
class Config:
    device: str
    compute_dtype: Literal["float32", "float16", "bfloat16"]
    train_until: Time
    val_every: Time | None
    save_every: Time | None
    save_on_train_end: bool
    batch_size: int


class Item(TypedDict):
    image: torch.FloatTensor
    label: int


class Batch(TypedDict):
    image: torch.FloatTensor
    label: torch.LongTensor


def train(
    exp: Experiment,
    train_data: Dataset[Item],
    val_data: Dataset[Item] | None,
    model: nn.Module,
    make_opt: Callable[[list[nn.Parameter]], Optimizer],
    cfg: Config,
):
    # "Infrastructure"
    device = torch.device(cfg.device)
    compute_dtype = getattr(torch, cfg.compute_dtype)
    autocast = lambda: torch.autocast(
        device.type,
        compute_dtype,
        enabled=cfg.compute_dtype != "float32",
    )

    # Model and optimizer setup
    model.to(device)
    opt = make_opt([*model.parameters()])
    if cfg.compute_dtype != "float32":
        opt = ScaledOptimizer(opt)

    # Time units and should_* flags
    step, epoch = 0, 0
    step_fns = {"step": lambda: step, "epoch": lambda: epoch}

    exp.register_step("step", lambda: step, default=True)
    exp.register_step("epoch", lambda: epoch)

    def make_until(x: Time):
        if isinstance(x, _Time):
            n, of = x.n, x.of
        else:
            n, of = x, "step"
        return cron.Until(step_fns[of], n)

    def make_every(x: Time | None, return_unit=False):
        if x is None:
            flag = cron.Never()
            of = "step"
        else:
            if isinstance(x, _Time):
                n, of = x.n, x.of
            else:
                n, of = x, "step"
            flag = cron.Every(step_fns[of], period=n)

        return flag, of

    should_train = make_until(cfg.train_until)
    should_val, val_time_unit = make_every(cfg.val_every)
    should_save, save_time_unit = make_every(cfg.save_every)

    # Data
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    def make_train_iter():
        nonlocal epoch
        while True:
            yield from train_loader
            epoch += 1

    train_iter = iter(make_train_iter())

    if val_data is not None:
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )
    else:
        val_loader = None

    def move_to_device(item: Batch) -> Batch:
        return {
            "image": item["image"].to(device),
            "label": item["label"].to(device=device, dtype=torch.long),
        }

    # Training loop

    def val_epoch():
        acc = None
        model.eval()

        for val_batch in val_loader:
            val_batch = move_to_device(val_batch)
            with torch.no_grad():
                with autocast():
                    logits: Tensor = model(val_batch["image"])
                    preds = F.softmax(logits, -1)

                if acc is None:
                    num_classes = logits.shape[-1]
                    acc = torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                    )
                    acc.to(device)

                acc.update(preds, val_batch["label"])

        model.train()

        exp.add_scalar("val/acc", acc.compute(), step=val_time_unit)

    def save_model(tag: str):
        dest = exp.dir / "ckpts" / f"model.{tag}.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_model(model, str(dest.absolute()))

    def train_step():
        batch = next(train_iter)
        batch = move_to_device(batch)

        with autocast():
            logits: Tensor = model(batch["image"])
            loss = F.cross_entropy(logits, batch["label"])

        if isinstance(opt, ScaledOptimizer):
            opt.step(loss)
        else:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        exp.add_scalar("train/loss", loss)

    pbar = exp.make_pbar()

    while should_train:
        if val_loader is not None:
            if should_val:
                val_epoch()

        if should_save:
            save_time = step_fns[save_time_unit]()
            save_model(tag=f"{save_time_unit}={save_time}")

        train_step()
        pbar.update()
        step += 1

    if cfg.save_on_train_end:
        save_model(tag="last")
