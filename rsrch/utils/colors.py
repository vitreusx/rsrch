from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np


def rgb2hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def hex2rgb(hex: str) -> tuple[int, int, int]:
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return r, g, b


@cache
def base_palette():
    src = Path(__file__).parent / "colors.npy"
    if src.exists():
        with open(src, "rb") as f:
            palette: np.ndarray = np.load(f)
    else:
        palette = _create_base_palette()
        with open(src, "wb") as f:
            np.save(f, palette)
    return palette


def _create_base_palette():
    """Create a general palette by starting from `#000000` and choosing the
    distant color, distance defined as Euclidean metric in RGB space."""

    from scipy.spatial.distance import cdist

    # Create grid of colors with spacing 32
    q = np.clip(np.arange(0, 257, 32), 0, 255)
    colors = np.stack(np.meshgrid(q, q, q), axis=-1).reshape(-1, 3)

    # Initialize palette as [#000000, #ffffff]
    palette = np.empty_like(colors)
    palette[0] = 0
    palette[1] = 255

    # Select consecutive colors by maximizing the distance to the current
    # palette
    for i in range(1, len(colors)):
        dists = cdist(colors, palette[:i])
        palette[i] = colors[np.argmax(dists.min(-1))]

    palette = palette.astype(np.uint8)
    return palette


@dataclass
class Palette:
    colors: np.ndarray
    ignore_color: tuple[int, int, int] | None

    def label2rgb(
        self,
        labels: np.ndarray,
        ignore_index: int | None = None,
    ):
        if ignore_index is None:
            return self.colors[labels]
        else:
            output = np.empty(
                shape=[*labels.shape, self.colors.shape[-1]],
                dtype=self.colors.dtype,
            )
            mask = labels != ignore_index
            if ignore_index == 0:
                # Reduce zero label
                output[mask] = self.colors[labels[mask] - 1]
            else:
                output[mask] = self.colors[labels[mask]]
            output[~mask] = self.ignore_color
            return output


def get_palette(
    num_classes: int,
    ignore_index: int | None = None,
) -> Palette:
    palette = base_palette()
    if ignore_index is not None:
        ignore_color = palette[1]  # White
        palette = np.concatenate((palette[:1], palette[2:]), axis=0)
    else:
        ignore_color = 0

    if num_classes > len(palette):
        raise ValueError("Palette size is too large.")

    return Palette(
        colors=palette[:num_classes],
        ignore_color=ignore_color,
    )
