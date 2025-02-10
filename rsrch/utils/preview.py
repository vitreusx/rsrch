import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


def make_grid(
    images: list[list[Image.Image]] | list[Image.Image],
    ncols: int | None = None,
    nrows: int | None = None,
    gap_px: int = 2,
) -> Image.Image:
    """Create a grid of images. The list of images can be 1-d or 2-d, in the former case the number of columns or rows must be provided in order to determine the shape of the grid."""

    if not isinstance(images[0], list):
        if nrows is not None:
            ncols = (len(images) + nrows - 1) // nrows
        elif ncols is not None:
            nrows = (len(images) + ncols - 1) // ncols
        else:
            raise ValueError("Either # of rows or columns must be provided.")
        images = [*images, *(None for _ in (nrows * ncols - len(images)))]
        grid = np.asarray(images, dtype=object)
        grid = grid.reshape((nrows, ncols))
    else:
        grid = np.asarray(images, dtype=object)
        nrows, ncols = grid.shape

    heights = [0 for _ in range(nrows)]
    widths = [0 for _ in range(ncols)]
    for row in range(nrows):
        for col in range(ncols):
            img = grid[row][col]
            if img is None:
                continue

            heights[row] = max(heights[row], img.height)
            widths[col] = max(widths[col], img.width)

    offset_x = np.cumsum(widths) - np.array(widths)
    offset_x += gap_px * np.arange(ncols)
    grid_w = offset_x[-1] + widths[-1]

    offset_y = np.cumsum(heights) - np.array(heights)
    offset_y += gap_px * np.arange(nrows)
    grid_h = offset_y[-1] + heights[-1]

    grid_img = Image.new("RGBA", (grid_w, grid_h))

    for row in range(nrows):
        for col in range(ncols):
            img = grid[row][col]
            if img is None:
                continue

            off_x = offset_x[col] + (widths[col] - img.width) // 2
            off_y = offset_y[row] + (heights[row] - img.height) // 2
            grid_img.paste(img, (off_x, off_y))

    return grid_img
