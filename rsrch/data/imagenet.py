from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image
from torch.utils import data

from rsrch.data.meta import ClsMeta


def _get_label_names(loc_synset_mapping_txt: str | Path):
    names = []
    with open(loc_synset_mapping_txt, "r") as f:
        for line in f:
            line = line.rstrip()
            defs = line[line.index(" ") + 1 :]
            pos = defs.find(",")
            if pos < 0:
                name = defs
            else:
                name = defs[:pos]
            names.append(name)
    return names


class ImageNet(data.Dataset):
    """ImageNet dataset.

    The dataset may also be a subset of IN-1k or any compatible one. Here's the
    required file structure:

    ```
    <root>/
    ├─ ILSVRC/
    │  ├─ Data/                   # Image files
    │  │  ├─ CLS-LOC/
    │  │  │  ├─ train/
    │  │  │  │  ├─ {wnid}/
    │  │  │  │  │  ├─ {image_file}
    │  │  │  │  │  ├─ ...
    │  │  │  ├─ val/
    │  │  │  │  ├─ {image_file}
    │  │  │  │  ├─ ...
    │  │  │  ├─ test/
    │  │  │  │  ├─ {image_file}
    │  │  │  │  ├─ ...
    │  ├─ ImageSets/              # Item lists
    │  │  ├─ CLS-LOC/
    │  │  │  ├─ train_cls.txt
    │  │  │  ├─ val.txt
    │  │  │  ├─ test.txt
    ├─ LOC_synset_mapping.txt     # A list of labels with WordNet IDs
    ├─ LOC_val_solution.csv       # An assignment of labels for val set
    ```
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = split
        self.img_root = self.root / "ILSVRC/Data/CLS-LOC" / split

        cls_lists = {"train": "train_cls.txt", "val": "val.txt", "test": "test.txt"}
        cls_list = self.root / "ILSVRC/ImageSets/CLS-LOC" / cls_lists[split]

        self._paths: list[Path] = []
        with open(cls_list, "r") as f:
            for line in f:
                path, index = line.strip().split(" ")
                assert int(index) - 1 == len(self._paths)
                self._paths.append(path)

        if split in ("train", "val"):
            wnid_to_label = {}
            with open(self.root / "LOC_synset_mapping.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    pos = line.index(" ")
                    wnid = line[:pos]
                    wnid_to_label[wnid] = len(wnid_to_label)

            if split == "train":
                self._labels = [
                    wnid_to_label[path.split("/")[0]] for path in self._paths
                ]
            elif split == "val":
                val_loc = pd.read_csv(self.root / "LOC_val_solution.csv")
                path_to_label = {}
                for _, row in val_loc.iterrows():
                    pred_s = row["PredictionString"]
                    wnid = pred_s[: pred_s.find(" ")]
                    label = wnid_to_label[wnid]
                    path_to_label[row["ImageId"]] = label
                self._labels = [path_to_label[p] for p in self._paths]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        img_path = (self.img_root / path).with_suffix(".JPEG")
        img = Image.open(img_path).convert("RGB")
        if self.split == "test":
            return img
        else:
            return {"image": img, "label": self._labels[idx]}

    def meta(self):
        loc_synset_mapping_txt = self.root / "LOC_synset_mapping.txt"
        label_names = _get_label_names(loc_synset_mapping_txt)
        classes = dict(enumerate(label_names))
        return ClsMeta({"classes": classes, "ignore_index": None})
