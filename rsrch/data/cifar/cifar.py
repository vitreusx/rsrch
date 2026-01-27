import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from ruamel.yaml import YAML

from .meta import ClsMeta


class CIFAR10:
    """CIFAR-10 dataset.

    File structure:
    ```
    <data_root>/
    └── cifar-10-batches-py/
        ├── data_batch_{1..5} # Train set
        └── test_batch        # Test set
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
    ):
        data_root = Path(data_root)

        batches = {
            "train": [f"data_batch_{idx}" for idx in range(1, 6)],
            "test": ["test_batch"],
        }[split]

        images, labels = [], []
        for fname in batches:
            with open(data_root / "cifar-10-batches-py" / fname, "rb") as f:
                batch = pickle.load(f, encoding="bytes")  # noqa: S301
            images.append(batch[b"data"])
            labels.extend(batch[b"label"])

        images = np.concatenate(images)
        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta():
        yaml = YAML(typ="safe", pure=True)
        with open(Path(__file__).parent / "cifar10.yml", "r") as f:
            data = yaml.load(f)
        return ClsMeta(data)


class CIFAR100:
    """CIFAR-100 dataset.

    File structure:
    ```
    <data_root>/
    └── cifar-100-python/
        ├── train          # Train set
        └── test           # Test set
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
    ):
        data_root = Path(data_root)

        with open(data_root / "cifar-10-batches-py" / split, "rb") as f:
            data = pickle.load(f, encoding="bytes")  # noqa: S301
            images, labels = data[b"data"], data[b"fine_labels"]

        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta():
        yaml = YAML(typ="safe", pure=True)
        with open(Path(__file__).parent / "cifar10.yml", "r") as f:
            data = yaml.load(f)

        return ClsMeta(
            {
                "classes": {
                    label: item["class"] for label, item in data["classes"].items()
                }
            }
        )
