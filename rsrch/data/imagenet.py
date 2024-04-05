from typing import Literal
from torch.utils import data
from pathlib import Path
import pandas as pd
from PIL import Image
import pickle
from hashlib import sha256


class ImageNet(data.Dataset):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        img_transform=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_root = self.root / "ILSVRC/Data/CLS-LOC" / split
        self.img_transform = img_transform

        cls_lists = {"train": "train_cls.txt", "val": "val.txt", "test": "test.txt"}
        cls_list = self.root / f"ILSVRC/ImageSets/CLS-LOC" / cls_lists[split]

        self._paths = []
        with open(cls_list, "r") as f:
            for line in f.readlines():
                self._paths.append(Path(line.split(" ")[0]))

        self.labels = {}

        wnid_to_label = {}
        with open(self.root / "LOC_synset_mapping.txt", "r") as f:
            for line in f.readlines():
                words = line.split(" ")
                wnid, desc = words[0], " ".join(words[1:]).strip()
                label = len(wnid_to_label)
                self.labels[label] = wnid, desc
                wnid_to_label[wnid] = label

        self._labels = None
        if split == "train":
            self._labels = {p.stem: wnid_to_label[p.parent.name] for p in self._paths}
        elif split == "val":
            self._labels = {}
            df = pd.read_csv(self.root / f"LOC_{split}_solution.csv")
            for _, row in df.iterrows():
                image_id = row["ImageId"]
                wnid = row["PredictionString"].split(" ")[0]
                self._labels[image_id] = wnid_to_label[wnid]

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx: int):
        r = {}
        path = self._paths[idx]
        img_path = (self.img_root / path).with_suffix(".JPEG")
        r["image"] = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            r["image"] = self.img_transform(r["image"])
        if self._labels is not None:
            r["label"] = self._labels[img_path.stem]
        return r


__all__ = ["ImageNet"]
