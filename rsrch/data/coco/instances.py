from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from rsrch.data.meta import cls_meta


class Box(NamedTuple):
    x: float
    y: float
    width: float
    height: float


class Instance(TypedDict):
    category: int
    bbox: Box
    iscrowd: bool | None
    mask: Image.Image


class COCOInstances(Sequence):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root).expanduser()
        self.split = split

        ann_path = self.root / f"annotations/instances_{split}2017.json"
        self.coco = COCO(ann_path)

        self.img_ids = self.coco.getImgIds()
        self.img_root = self.root / f"{split}2017"

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path)

        ann_ids = self.coco.getAnnIds(img_id)
        coco_anns = self.coco.loadAnns(ann_ids)

        instances: list[Instance] = []
        for coco_ann in coco_anns:
            mask = self.coco.annToMask(coco_ann)
            mask = Image.fromarray((255 * mask).astype(np.uint8))
            instances.append(
                {
                    "category": coco_ann["category_id"],
                    "bbox": Box(coco_ann["bbox"]),
                    "iscrowd": coco_ann["iscrowd"] > 0,
                    "mask": mask,
                }
            )

        return {"image": img, "instances": instances}

    def _get_meta(self):
        classes = {}
        categories = self.coco.loadCats(self.coco.getCatIds())
        for cat in categories:
            classes[cat["id"]] = cat["name"]

        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta():
        return cls_meta(Path(__file__).parent / "coco.yml")
