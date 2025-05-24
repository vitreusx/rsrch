from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image

from .meta import seg_meta


class VOCSegmentation(Sequence):
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        super().__init__()
        self.root = Path(root).expanduser() / "VOCdevkit/VOC2012"
        self.split = split

        image_set_path = self.root / "ImageSets/Segmentation" / f"{split}.txt"
        with open(image_set_path, "r") as f:
            self._ids = [line.rstrip() for line in f]

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, index: int):
        id = self._ids[index]
        img_path = self.root / "JPEGImages" / f"{id}.jpg"
        image = Image.open(img_path)
        seg_map_path = self.root / "SegmentationClass" / f"{id}.png"
        seg_map = Image.open(seg_map_path)
        return {"image": image, "labels": seg_map}

    @staticmethod
    def meta():
        return seg_meta(Path(__file__).parent / "voc.yml")
