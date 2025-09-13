from functools import cached_property
from pathlib import Path
from typing import TypedDict

import numpy as np
from ruamel.yaml import YAML

from rsrch.utils.colors import hex2rgb

from .utils import is_contiguous, typed_cache


class PanopticLabelInfo(TypedDict):
    name: str
    isthing: bool
    palette: str | tuple[int, int, int] | None


class PanopticMetaData(TypedDict):
    classes: dict[int, str | PanopticLabelInfo]
    ignore_index: int | None


class PanopticMeta:
    def __init__(self, data: PanopticMetaData):
        self.data = data

    @cached_property
    def label_to_name(self) -> dict[str, str]:
        map_ = {}
        for label, info in self.data["classes"].items():
            name = info if isinstance(info, str) else info["name"]
            map_[label] = name

        if not is_contiguous([*map_]):
            raise RuntimeError("Class labels must be contiguous")

        return map_

    def is_stuff(self, label: int) -> bool:
        return self.data["classes"][label]["isthing"]

    @cached_property
    def num_classes(self) -> int:
        return len(self.label_to_name)

    @cached_property
    def names(self) -> list[str]:
        map_ = self.label_to_name
        return [map_[label] for label in map_]

    @cached_property
    def label_to_color(self) -> dict[int, tuple[int, int, int]] | None:
        map_ = {}
        for label, info in self.data["classes"].items():
            if not isinstance(info, dict) or info.get("palette") is None:
                return None
            color = info["palette"]
            if isinstance(color, str):
                color = hex2rgb(color)
            map_[label] = color

        if not is_contiguous([*map_]):
            raise ValueError("Labels must be contiguous")

        return map_

    @cached_property
    def palette(self) -> np.ndarray | None:
        map_ = self.label_to_color
        if map_ is None:
            return None

        palette = np.array(map_[label] for label in map_)
        palette = palette.astype(np.uint8)
        return palette

    @property
    def ignore_index(self) -> int | None:
        return self.data.get("ignore_index", None)


@typed_cache
def panoptic_meta(meta_yml: str | Path):
    yaml = YAML(typ="safe", pure=True)
    with open(meta_yml.expanduser(), "r") as f:
        data = yaml.load(f)

    return PanopticMeta(data)
