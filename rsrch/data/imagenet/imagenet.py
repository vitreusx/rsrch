from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

from rsrch.data.meta import ClsMeta


def parse_loc_synset_mapping(path: str | Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            pos = line.index(" ")
            wnid, defs = line[:pos], line[pos + 1 :]
            pos = defs.find(",")
            name = defs if pos < 0 else defs[:pos]
            records.append((wnid, name, defs))

    return pd.DataFrame.from_records(
        records,
        columns=["wnid", "name", "defs"],
    )


def _get_label_names(loc_synset_mapping_txt: str | Path):
    names = []
    with open(loc_synset_mapping_txt, "r") as f:
        for line in f:
            line = line.rstrip()
            defs = line[line.index(" ") + 1 :]
            pos = defs.find(",")
            name = defs if pos < 0 else defs[:pos]
            names.append(name)
    return names


class ImageNet:
    """ImageNet dataset.

    The dataset may also be a subset of IN-1k or any compatible one.

    File structure:
    ```
    <data_root>/
    ├── ILSVRC/
    │   ├── Annotations/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.xml
    │   │   └── val/
    │   │       └── {img_id}.xml
    │   ├── Data/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.JPEG
    │   │   ├── val/
    │   │   │   └── {img_id}.JPEG
    │   │   └── test/
    │   │       └── {img_id}.JPEG
    │   └── ImageSets/CLS-LOC/
    │       ├── train_cls.txt
    │       ├── train_loc.txt
    │       ├── val.txt
    │       └── test.txt
    └── LOC_synset_mapping.txt     # A list of labels with WordNet IDs
    └── LOC_train_solution.csv
    └── LOC_val_solution.csv
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
        self.ann_root = self.root / "ILSVRC/Annotations/CLS-LOC" / split

        cls_lists = {"train": "train_cls.txt", "val": "val.txt", "test": "test.txt"}
        cls_list = self.root / "ILSVRC/ImageSets/CLS-LOC" / cls_lists[split]

        self.paths: list[str] = []
        with open(cls_list, "r") as f:
            for line in f:
                path, index = line.strip().split(" ")
                if int(index) - 1 != len(self.paths):
                    raise RuntimeError("Invalid class order")
                self.paths.append(path)

        synset_df = parse_loc_synset_mapping(self.root / "LOC_synset_mapping.txt")
        self.wnid_to_label = {
            wnid: label for label, wnid in enumerate(synset_df["wnid"])
        }

        if split == "train":
            self.wnids = []
            for path in self.paths:
                wnid = path.split("/")[0]
                self.wnids.append(wnid)
        elif split == "val":
            # We get the IDs from the solution file, because it's faster than
            # parsing all the XML files
            sol_df = pd.read_csv(self.root / "LOC_val_solution.csv")
            wnids_map = {}
            for _, row in sol_df.iterrows():
                pred: str = row["PredictionString"]
                wnid = pred[: pred.find(" ")]
                wnids_map[row["ImageId"]] = wnid
            self.wnids = [wnids_map[path] for path in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_path = self.img_root / (path + ".JPEG")
        img = Image.open(img_path).convert("RGB")

        if self.split in ("train", "val"):
            wnid = self.wnids[idx]
            label = self.wnid_to_label[wnid]
            return {"image": img, "label": label}
        else:
            return img

    def meta(self):
        loc_synset_mapping_txt = self.root / "LOC_synset_mapping.txt"
        synset_df = parse_loc_synset_mapping(loc_synset_mapping_txt)
        label_names = synset_df["name"]
        classes = dict(enumerate(label_names))
        return ClsMeta({"classes": classes, "ignore_index": None})
