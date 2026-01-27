from functools import cached_property
from pathlib import Path
from typing import TypedDict

import numpy as np
from ruamel.yaml import YAML

from .colors import Palette, get_palette, hex2rgb
from .utils import is_contiguous, typed_cache


class SegLabelInfo(TypedDict):
    name: str
    palette: str | tuple[int, int, int] | None


class SegMetaData(TypedDict):
    classes: dict[int, str | SegLabelInfo]
    ignore_index: int | None


class SegMeta:
    def __init__(self, data: SegMetaData):
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
            raise ValueError("Labels' list must be contiguous")
        if "ignore_color" not in self.data:
            raise ValueError("Need to provide ignore_index")
        return map_

    @cached_property
    def palette(self) -> Palette:
        color_map = self.label_to_color
        if color_map is not None:
            colors = np.array(color_map[label] for label in color_map)
            colors = colors.astype(np.uint8)
            ignore_color = np.array(self.data["ignore_color"])
            return Palette(colors, ignore_color)
        else:
            return get_palette(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )

    @property
    def ignore_index(self) -> int | None:
        return self.data.get("ignore_index", None)


@typed_cache
def seg_meta(meta_yml: str | Path):
    yaml = YAML(typ="safe", pure=True)
    with open(meta_yml, "r") as f:
        data = yaml.load(f)

    return SegMeta(data)
