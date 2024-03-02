from numbers import Number
from pathlib import Path
from typing import Callable

from moviepy.editor import VideoClip


class Experiment:
    project: str
    name: str
    dir: Path

    def register_step(
        self,
        tag: str,
        value_fn: Callable[[], Number],
        default: bool,
    ):
        """Add a step metric. Afterwards, one can use the tag name instead of
        the step value in logging functions. Moreover, one can declare the
        step metric to be a default one, which allows for not having to pass
        the step explicitly in logging functions.
        """

    def add_scalar(
        self,
        tag: str,
        value: Number,
        step: str | Number | None,
    ):
        """Log a scalar value.
        :param tag: Tag for the scalar value.
        :param y: Value to log.
        :param step: Either a step value, a step tag, as declared in `add_step`,
        or None, in which case the default step is used.
        """

    def add_video(
        self,
        tag: str,
        vid: VideoClip,
        fps: float,
        step: str | Number | None,
    ):
        """Log a video clip."""
