from pathlib import Path
from PIL import Image
from moviepy.editor import VideoClip
from numbers import Number
from typing import TypeAlias
from abc import ABC, abstractmethod

Step: TypeAlias = int | str | None


class Board(ABC):
    @abstractmethod
    def add_config(self, config: dict):
        ...

    @abstractmethod
    def log(self, message: str):
        ...

    @abstractmethod
    def register_step(self, name: str, value_fn, default=False):
        ...

    @abstractmethod
    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        ...

    @abstractmethod
    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        ...

    @abstractmethod
    def add_video(self, tag: str, vid: VideoClip, *, fps=30.0, step: Step = None):
        ...
