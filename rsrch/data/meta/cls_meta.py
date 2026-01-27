from dataclasses import make_dataclass
from functools import cached_property
from pathlib import Path
from pprint import pformat
from typing import TypedDict

from ruamel.yaml import YAML

from .utils import is_contiguous, typed_cache


class ClsMetaData(TypedDict):
    classes: dict[int, str]
    ignore_index: int | None


class ClsMeta:
    """Metadata for classification datasets."""

    def __init__(self, data: ClsMetaData):
        self.data = data
        if not is_contiguous([*data["classes"]]):
            raise RuntimeError("Class labels must be contiguous")

    @property
    def label_to_name(self):
        return self.data["classes"]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_name)

    @cached_property
    def names(self) -> list[str]:
        map_ = self.label_to_name
        return [map_[label] for label in map_]

    @property
    def ignore_index(self) -> int | None:
        return self.data.get("ignore_index", None)

    def __repr__(self):
        cls = make_dataclass(
            self.__class__.__name__,
            ["names", "ignore_index"],
        )
        return pformat(cls(self.names, self.ignore_index))


@typed_cache
def cls_meta(meta_yml: str | Path):
    """Load metadata for image classification from YAML file."""
    yaml = YAML(typ="safe", pure=True)
    with open(meta_yml, "r") as f:
        data = yaml.load(f)

    return ClsMeta(data)
