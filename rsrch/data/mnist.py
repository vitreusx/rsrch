from pathlib import Path
from typing import Literal

from torchvision import datasets

from .meta import ClsMeta


class MNIST:
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"],
        download: bool = False,
    ):
        self.base = datasets.MNIST(
            root=root,
            train=(split == "train"),
            download=download,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta():
        digits = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        return ClsMeta(
            {
                "classes": {digit: name for digit, name in enumerate(digits)},
                "ignore_index": None,
            }
        )
