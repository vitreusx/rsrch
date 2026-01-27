from typing import Any, TypedDict


class Info(TypedDict):
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str


class License(TypedDict):
    url: str
    id: int
    name: str


class Image(TypedDict):
    license: int
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    id: int


class DetectAnn(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: Any
    area: float
    bbox: tuple[int, int, int, int]
    iscrowd: int


class DetectCategory(TypedDict):
    supercategory: str
    id: int
    name: str


class DetectAnnFile(TypedDict):
    info: Info
    licenses: list[License]
    images: list[Image]
    annotations: list[DetectAnn]
    categories: list[DetectCategory]


class SegmentInfo(TypedDict):
    id: int
    category_id: int
    area: int
    bbox: tuple[int, int, int, int]
    iscrowd: int


class PanopticAnn(TypedDict):
    image_id: int
    file_name: str
    segments_info: list[SegmentInfo]


class PanopticCategory(TypedDict):
    id: int
    name: str
    supercategory: str
    isthing: int
    color: tuple[int, int, int]


class PanopticAnnFile(TypedDict):
    info: Info
    licenses: list[License]
    images: list[Image]
    annotations: list[PanopticAnn]
    categories: list[PanopticCategory]
