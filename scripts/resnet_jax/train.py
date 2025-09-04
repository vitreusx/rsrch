from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypedDict, TypeVar

import albumentations as A
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from PIL import Image
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

from rsrch.data.imagenet import ImageNet
from rsrch.data.mnist import MNIST
from rsrch.exp import boards
from rsrch.exp.experiment import Experiment
from rsrch.utils import cron, repro
from rsrch.utils.cast import cast

# isort: off
import models
from config import Config, TimeDelta
# isort: on


class Item(TypedDict):
    """A dataset item for image classification."""

    image: Array  # (C, H, W), dtype: float
    label: int


class Batch(TypedDict):
    """A batch of items for image classification."""

    image: Array  # (N, C, H, W), dtype: float
    label: Array  # (N), dtype: int32


class ToArray(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(p=1.0)

    def get_params_dependent_on_data(self, params, data):
        return {}

    def apply(self, image: np.ndarray, **kwargs):
        array = jnp.asarray(image)
        if len(array.shape) == 2:
            array = jnp.expand_dims(array, 0)
        else:
            array = jnp.moveaxis(array, -1, 0)
        return array


class Dataset:
    def __init__(
        self,
        base: ImageNet,
        transforms: list[A.ImageOnlyTransform] | None = None,
        subset: list[int] | None = None,
        mean: float | tuple[float] = (0.485, 0.456, 0.406),
        std: float | tuple[float] = (0.229, 0.224, 0.225),
    ):
        self.base = base
        # Metadata (ignore index, # of classes etc.) for the dataset
        self.meta = base.meta()
        self.mean, self.std = mean, std

        if subset is None:
            self.indices = range(len(self.base))
        else:
            self.indices = subset

        if transforms is None:
            transforms = ()

        self.img_transform = A.Compose(
            [
                *transforms,
                A.Normalize(self.mean, self.std),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Item:
        item = self.base[self.indices[index]]
        image_nd = np.asarray(item["image"])
        item["image"] = self.img_transform(image=image_nd)["image"]
        return item

    def to_pil_image(self, image: Array):
        image = jnp.moveaxis(image, 0, -1)  # [C, H, W] -> [H, W, C]
        mean = jnp.array(self.mean, device=image.device)
        std = jnp.array(self.std, device=image.device)
        image = image * std + mean  # Invert the normalization transform
        image = jnp.clip(255 * image, 0.0, 255.0).astype(jnp.uint8)
        image = Image.fromarray(image)
        return image

    @staticmethod
    def collate_fn(batch: list[Item]) -> Batch:
        image = jnp.stack([item["image"] for item in batch])
        if len(image.shape) == 3:
            image = jnp.expand_dims(image, 1)  # [N, H, W] -> [N, 1, H, W]
        else:
            image = jnp.moveaxis(image, -1, 1)  # [N, H, W, C] -> [N, C, H, W]

        label = jnp.array(
            [item["label"] for item in batch],
            dtype=jnp.int32,
        )

        return {"image": image, "label": label}


@eqx.filter_jit
def train_step(
    model: models.Resnet,
    state: eqx.nn.State,
    opt: optax.GradientTransformation,
    opt_state: Any,
    input: Array,
    labels: Array,
):
    @partial(eqx.filter_value_and_grad, has_aux=True)
    @jax.named_scope("forward_pass")
    def compute_loss(
        model: models.Resnet,
        state: eqx.nn.State,
    ):
        batch_model = jax.vmap(
            model,
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )
        logits, new_state = batch_model(input, state)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return losses.mean(), new_state

    (loss, new_state), grads = compute_loss(model, state)

    with jax.named_scope("opt_step"):
        updates, new_opt_state = opt.update(grads, opt_state, model)
        new_model: models.Resnet = eqx.apply_updates(model, updates)

    return new_model, new_state, new_opt_state, loss


@eqx.filter_jit
def val_step(val_model: models.Resnet, state: eqx.nn.State, input: Array):
    batch_model = jax.vmap(
        val_model,
        axis_name="batch",
        in_axes=(0, None),
        out_axes=(0, None),
    )
    logits, _ = batch_model(input, state)
    return logits


class AccState(TypedDict):
    count: int
    total: int


class Accuracy:
    def __init__(self, num_classes: int, top_k: int = 1):
        self.num_classes = num_classes
        self.top_k = top_k

    def init(self) -> AccState:
        return {"count": 0, "total": 0}

    def update(self, state: AccState, logits: Array, labels: Array) -> AccState:
        topk = jnp.argsort(logits, axis=-1, descending=True)[:, : self.top_k]
        correct = (topk == jnp.expand_dims(labels, -1)).any(-1)
        return {
            "count": state["count"] + correct.sum(),
            "total": state["total"] + correct.size,
        }

    def compute(self, state: AccState) -> float:
        return state["count"] / state["total"]


class Trainer:
    project = "resnet_jax"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self):
        self.setup_infra()
        self.setup_data()
        self.setup_loaders()
        self.setup_model()
        self.setup_prof()

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

        should_run = get_flag(self.cfg.train_for, mode="until")
        should_val = get_flag(self.cfg.val_every)
        self.should_log = get_flag(self.cfg.log_every)
        # self.should_save_samples = cron.Once()
        # self.should_save_val_samples = cron.Once()

        # Training loop
        self.pbar = self.exp.make_pbar(desc="Train loop")
        while should_run:
            if should_val:
                self.val_epoch()
            self.train_step()
            self.step += 1
            self.pbar.update()

    def setup_infra(self):
        repro.seed_all(self.cfg.seed)
        self.exp = Experiment(
            project=self.project,
            create_commit=self.cfg.create_exp_commit,
        )
        self.exp.add_board(boards.Tensorboard(self.exp.dir / "board", launch=True))

        self.step, self.epoch = 0, 0
        self.exp.register_step("step", lambda: self.step, default=True)
        self.exp.register_step("epoch", lambda: self.epoch)

        self.device = jax.devices()[0]

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
            val_idxes = np.random.choice(len(val_ds), size=val_size, replace=False)
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
            val_idxes = np.random.choice(len(val_ds), size=val_size, replace=False)
            val_subset = val_idxes.tolist()
        else:
            val_subset = None

        self.val_data = Dataset(val_ds, subset=val_subset, **kw)

    def setup_model(self):
        make_model = getattr(models, self.cfg.model)
        key = jax.random.key(seed=self.cfg.seed)
        self.model, self.state = eqx.nn.make_with_state(make_model)(
            in_channels=self.in_channels,
            num_classes=self.meta.num_classes,
            key=key,
        )

        self.opt = optax.adamw(3e-4)
        params = eqx.filter(self.model, eqx.is_inexact_array)
        self.opt_state = self.opt.init(params)

    def setup_loaders(self):
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_data.collate_fn,
        )

        self.train_iter = self.get_train_iter()

        val_batch_size = self.cfg.val_batch_size or self.cfg.batch_size
        self.val_loader = DataLoader(
            self.val_data,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.val_data.collate_fn,
        )

    def setup_prof(self):
        self.is_profiling = False
        if self.cfg.profile:
            self.should_profile = cron.If(lambda: 128 <= self.step < 256)
        else:
            self.should_profile = cron.Never()

    def get_train_iter(self):
        self.epoch = 0
        while True:
            yield from self.train_loader
            self.epoch += 1

    def train_step(self):
        if self.should_profile:
            if not self.is_profiling:
                jax.profiler.start_trace(self.exp.dir / "board")
                self.exp.info("Starting profiling")
                self.is_profiling = True
            loss = self._train_step_prof()
        else:
            if self.is_profiling:
                jax.profiler.stop_trace()
                self.exp.info("Ended profiling")
                self.is_profiling = False
            loss = self._train_step()

        if self.should_log:
            self.exp.add_scalar("train/loss", loss)

    def _train_step_prof(self):
        with jax.profiler.StepTraceAnnotation("train_step"):
            with jax.profiler.TraceAnnotation("load_data"):
                batch = next(self.train_iter)
                jax.block_until_ready(batch)

            self.model, self.state, self.opt_state, loss = train_step(
                model=self.model,
                state=self.state,
                opt=self.opt,
                opt_state=self.opt_state,
                input=batch["image"],
                labels=batch["label"],
            )
            jax.block_until_ready(self.model)

        return loss

    def _move_to_device(self, item: dict):
        result = {}
        for k, v in item.items():
            if isinstance(v, jax.Array):
                v_dev = jax.device_put(v, self.device)
            else:
                v_dev = v
            result[k] = v_dev
        return result

    def _train_step(self):
        batch = next(self.train_iter)
        self.model, self.state, self.opt_state, loss = train_step(
            model=self.model,
            state=self.state,
            opt=self.opt,
            opt_state=self.opt_state,
            input=batch["image"],
            labels=batch["label"],
        )
        return loss

    def val_epoch(self):
        top1 = Accuracy(self.meta.num_classes, top_k=1)
        top1_state = top1.init()

        if self.meta.num_classes > 5:
            top5 = Accuracy(self.meta.num_classes, top_k=5)
            top5_state = top5.init()
        else:
            top5 = None

        val_model = eqx.nn.inference_mode(self.model)

        for batch in self.val_loader:
            logits = val_step(val_model, self.state, batch["image"])
            top1_state = top1.update(top1_state, logits, batch["label"])
            if top5 is not None:
                top5_state = top5.update(top5_state, logits, batch["label"])

        top1_v = top1.compute(top1_state)
        val_unit = self.cfg.val_every.of
        self.exp.add_scalar("val/acc", top1_v, step=val_unit)
        if top5 is not None:
            top5_v = top5.compute(top5_state)
            self.exp.add_scalar("val/acc_top5", top5_v, step=val_unit)


def main():
    yaml = YAML(typ="safe", pure=True)
    with open(Path(__file__).parent / "config.yml", "r") as f:
        cfg = cast(yaml.load(f), Config)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
