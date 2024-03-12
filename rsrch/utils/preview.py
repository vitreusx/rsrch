import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


def make_grid(images, ncols=None, padding=2) -> Image.Image:
    def _convert(img):
        if not isinstance(img, Image.Image):
            img = F.to_pil_image(img)
        return img.convert("RGBA")

    def flat_to_2d():
        assert ncols is not None
        nrows = int(np.ceil(len(images) / ncols))
        grid = []
        for row_idx in range(nrows):
            row = []
            for col_idx in range(ncols):
                idx = row_idx * ncols + col_idx
                img = _convert(images[idx]) if idx < len(images) else None
                row.append(img)
            grid.append(row)
        return grid

    # Here we convert images to 2d grid
    if isinstance(images, torch.Tensor):
        grid = flat_to_2d()
    elif isinstance(images, list):
        if isinstance(images[0], list):
            grid = [[_convert(img) for img in row] for row in images]
        else:
            grid = flat_to_2d()
    else:
        grid = [[_convert(images)]]

    nrows, ncols = len(grid), len(grid[0])

    heights = [None for _ in range(nrows)]
    widths = [None for _ in range(ncols)]
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img = grid[row_idx][col_idx]
            if img is not None:
                w, h = img.size
                if widths[col_idx] is None:
                    widths[col_idx] = w
                else:
                    assert widths[col_idx] == w
                if heights[row_idx] is None:
                    heights[row_idx] = h
                else:
                    assert heights[row_idx] == h

    for row_idx in range(nrows):
        if heights[row_idx] is None:
            heights[row_idx] = 0

    for col_idx in range(ncols):
        if widths[col_idx] is None:
            widths[col_idx] = 0

    total_width = sum(widths) + padding * (len(widths) - 1)
    total_height = sum(heights) + padding * (len(heights) - 1)
    grid_img = Image.new("RGBA", (total_width, total_height))

    offset_x = np.cumsum([0, *widths]) + padding * np.arange(len(widths) + 1)
    offset_y = np.cumsum([0, *heights]) + padding * np.arange(len(heights) + 1)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img = grid[row_idx][col_idx]
            if img is None:
                w, h = widths[col_idx], heights[row_idx]
                img = Image.new("RGBA", (w, h), 0)

            off_x, off_y = offset_x[col_idx], offset_y[row_idx]
            grid_img.paste(img, (off_x, off_y))

    return grid_img
