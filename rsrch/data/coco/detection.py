from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

from PIL import Image
from pycocotools.coco import COCO

from rsrch.data.meta import cls_meta


class Box(NamedTuple):
    x: float
    y: float
    width: float
    height: float


class Detection(TypedDict):
    category: int
    bbox: Box
    iscrowd: bool | None


class COCODetection(Sequence):
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

        detections = [
            {
                "category": coco_ann["category_id"],
                "bbox": Box(coco_ann["bbox"]),
                "iscrowd": coco_ann["iscrowd"] > 0,
            }
            for coco_ann in self.coco.loadAnns(ann_ids)
        ]

        return {"image": img, "objects": detections}

    @staticmethod
    def meta():
        return cls_meta(Path(__file__).parent / "coco.yml")
