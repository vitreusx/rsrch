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
        """Add a config value as a dict."""
        pass

    def log(self, level: int, message: str):
        """Log a message."""
        pass

    def register_step(self, name: str, value_fn, default=False):
        """Register a step value. Afterwards, in add_* functions, one can use string value for `step` to reference the given step."""
        pass

    def set_as_default(self, name: str):
        """Set a registered step value as a default one. Thereafter, if `step` parameter is not supplied to an add_* function, the default one is used."""
        pass

    def add_scalar(self, tag: str, value: Number, *, step: Step = None):
        """Add a scalar value. Accepts numbers (also 0-d tensors.)"""
        pass

    def add_image(self, tag: str, image: Image.Image, *, step: Step = None):
        """Add an image. Accepts PIL images."""
        pass

    def add_video(self, tag: str, vid: VideoClip, *, step: Step = None):
        """Add a video. Accepts moviepy clips."""
        pass

    def add_dict(self, tag: str, value: dict, *, step: Step = None):
        """Add a dictionary."""
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

    def set_as_default(self, name: str):
        self._def_step = name

    def _get_step(self, step: int | str | None = None):
        if step is None:
            return self._steps[self._def_step]()
        elif isinstance(step, str):
            return self._steps[step]()
        else:
            return step
