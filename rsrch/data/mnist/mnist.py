import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


def load_idx(fp):
    """Load array in IDX format."""

    # First four bytes: two zeros, dtype and num of axes
    # Note: The values are all encoded in big-endian
    magic = fp.read(4)
    dtype = {
        0x08: np.dtype(">u1"),
        0x09: np.dtype(">i1"),
        0x0B: np.dtype(">i2"),
        0x0C: np.dtype(">i4"),
        0x0D: np.dtype(">f4"),
        0x0E: np.dtype(">f8"),
    }[magic[2]]
    num_axes = int(magic[3])

    dims = np.frombuffer(fp.read(num_axes * 4), np.dtype(">i4"))

    nbytes = fp.read(dtype.itemsize * math.prod(dims))
    return np.frombuffer(nbytes, dtype).reshape(*dims)


class MNIST:
    """MNIST dataset.

    File structure:
    ```
    <data_root>/
    ├── train-labels-idx1-ubyte
    ├── t10k-labels-idx1-ubyte
    ├── train-images-idx3-ubyte
    └── t10k-images-idx3-ubyte
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
    ):
        data_root = Path(data_root)
        prefix = {"train": "train", "test": "t10k"}[split]

        with open(data_root / f"{prefix}-images-idx3-ubyte", "rb") as f:
            self.images = load_idx(f)

        with open(data_root / f"{prefix}-labels-idx1-ubyte", "rb") as f:
            self.labels = load_idx(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: str):
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        return {"image": image, "label": label}
