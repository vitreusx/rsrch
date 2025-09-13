import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from rsrch.data.meta import seg_meta

if TYPE_CHECKING:
    from .utils.schema import DetectAnnFile


class COCOStuff(Sequence):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root).expanduser()
        self.split = split

        ann_path = self.root / f"annotations/stuff_{split}2017.json"
        with open(ann_path, "r") as f:
            self.ann_file: DetectAnnFile = json.load(f)

        self.img_root = self.root / f"{split}2017"
        self.seg_root = self.root / f"annotations/stuff_{split}2017_pixelmaps"

    def __len__(self):
        return len(self.ann_file["images"])

    def __getitem__(self, index: int):
        img_info = self.ann_file["images"][index]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path)

        seg_map_path = self.seg_root / f"{img_path.stem}.png"
        seg_map = Image.open(seg_map_path)

        return {"image": img, "labels": seg_map}

    def _compute_meta(self):
        classes = {}
        for cat in self.ann_file["categories"]:
            classes[cat["id"]] = {
                "name": cat["name"],
                "supercategory": cat["supercategory"],
            }

        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta():
        return seg_meta(Path(__file__).parent / "coco_stuff.yml")
