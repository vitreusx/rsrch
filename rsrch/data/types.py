from typing import NamedTuple, TypedDict

import numpy as np
from PIL import Image


class ClsItem(TypedDict):
    image: Image.Image
    label: int


class SegItem(TypedDict):
    image: Image.Image
    labels: Image.Image


class Box(NamedTuple):
    x: float
    y: float
    width: float
    height: float


class Detection(TypedDict):
    category: int
    bbox: tuple | Box


class DetectItem(TypedDict):
    image: Image.Image
    objects: list[Detection]


class Instance(TypedDict):
    category: int
    mask: Image.Image


class InstanceItem(TypedDict):
    image: Image.Image
    instances: list[Detection]


class Segment(TypedDict):
    id: int
    category: int


class PanopticItem(TypedDict):
    image: Image.Image
    ids: np.ndarray
    segments: list[Segment]
