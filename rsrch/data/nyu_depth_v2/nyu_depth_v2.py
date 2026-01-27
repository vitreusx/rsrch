from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from rsrch.data.meta import seg_meta


class NYUDepthV2(Sequence):
    """NYU Depth V2 dataset."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        mat_file = self.root / "nyu_depth_v2_labeled.mat"
        self._f = h5py.File(mat_file)

    def __len__(self):
        return int(self._f["images"].shape[0])

    def __getitem__(self, index: int):
        image_nd = self._f["images"][index]
        image = Image.fromarray(image_nd, mode="RGB")
        seg_map_nd = self._f["labels"]
        seg_map = Image.fromarray(seg_map_nd, mode="I;16")
        return {"image": image, "labels": seg_map}

    def _compute_meta(self):
        names = ["unlabeled"]  # Label zero is ignored
        for ref in self._f["names"][0]:
            value_ds = self._f[ref]
            name = np.asarray(value_ds).astype(np.uint8).tobytes().decode("utf-8")
            names.append(name)

        classes = {label: name for label, name in enumerate(names)}
        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta():
        return seg_meta(Path(__file__).parent / "nyu_depth_v2.yml")
