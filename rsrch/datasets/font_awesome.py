import torch
import torch.utils.data as data
import os
from pathlib import Path
from dataclasses import dataclass
from PIL import Image


@dataclass
class Item:
    name: str
    image: Image.Image


class FontAwesome(data.Dataset[Item]):
    def __init__(self, root: os.PathLike, color: str, resolution: int):
        super().__init__()
        self.root = Path(root)
        self.color = color
        self.resolution = resolution

        self._image_dir = self.root / color / "png" / f"{resolution}"
        self._image_paths = [*self._image_dir.iterdir()]

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        path = self._image_paths[idx]
        name = path.with_suffix("").name
        image = Image.open(path)
        return Item(name=name, image=image)
