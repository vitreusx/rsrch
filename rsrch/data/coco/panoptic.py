import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from rsrch.data.meta import panoptic_meta

if TYPE_CHECKING:
    from .utils.schema import PanopticAnnFile


class COCOPanoptic(Sequence):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root).expanduser()
        self.split = split

        ann_file = self.root / f"annotations/panoptic_{split}2017.json"
        with open(ann_file, "r") as f:
            self.ann_file: PanopticAnnFile = json.load(f)

        self.img_anns = {image["id"]: [] for image in self.ann_file["images"]}
        for ann_id, ann in enumerate(self.ann_file["annotations"]):
            self.img_anns[ann["image_id"]].append(ann_id)

        self.img_root = self.root / f"{self.split}2017"
        self.ann_root = self.root / f"annotations/panoptic_{self.split}2017"

        if any(len(v) != 1 for v in self.img_anns.values()):
            raise RuntimeError("Need to have 1 annotation per image")

    def __len__(self):
        return len(self.ann_file["images"])

    def __getitem__(self, index: int):
        img_info = self.ann_file["images"][index]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path)

        img_id = img_info["id"]
        ann_id: int = self.img_anns[img_id][0]
        ann = self.ann_file["annotations"][ann_id]
        ann_img_path = self.ann_root / ann["file_name"]
        ann_img = Image.open(ann_img_path)
        ids = np.asarray(ann_img).astype(np.int32)
        ids = (ids * np.array([1, 256, 256**2])).sum(-1)

        return {
            "image": img,
            "ann_img": ann_img,
            "ids": ids,
            "segments": ann["segments_info"],
        }

    def _get_meta(self):
        classes = {}
        for cat in self.ann_file["categories"]:
            classes[cat["id"]] = {
                "name": cat["name"],
                "isthing": cat["isthing"] > 0,
                "supercategory": cat["supercategory"],
            }

        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta():
        return panoptic_meta(Path(__file__).parent / "coco_panoptic.yml")
