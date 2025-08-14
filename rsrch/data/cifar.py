from pathlib import Path
from typing import Literal

from ruamel.yaml import YAML
from torchvision import datasets

from .meta import ClsMeta


class CIFAR10:
    """[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"],
    ):
        self.base = datasets.CIFAR10(root, train=(split == "train"))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta():
        yaml = YAML(typ="safe", pure=True)
        with open(Path(__file__).parent / "cifar10.yml", "r") as f:
            data = yaml.load(f)
        return ClsMeta(data)


class CIFAR100:
    """[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"],
    ):
        self.base = datasets.CIFAR100(root, train=(split == "train"))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
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
