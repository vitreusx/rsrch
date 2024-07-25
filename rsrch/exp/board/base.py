import logging
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import TypeAlias

from moviepy.editor import VideoClip
from PIL import Image

Step: TypeAlias = int | str | None


class Board:
    def add_config(self, config: dict):
        pass

    def log(self, level: int, message: str):
        pass

    def register_step(self, name: str, value_fn, default=False):
        pass

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        pass

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        pass

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        pass


class StepMixin:
    def __init__(self):
        super().__init__()
        self._steps = {}
        self._def_step = None

    def register_step(self, name: str, value_fn, default=False):
        self._steps[name] = value_fn
        if default:
            self._def_step = name

    def _get_step(self, step: int | str | None = None):
        if step is None:
            return self._steps[self._def_step]()
        elif isinstance(step, str):
            return self._steps[step]()
        else:
            return step
