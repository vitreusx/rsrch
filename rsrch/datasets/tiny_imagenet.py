import torch
import rsrch.utils.data as data
import os
from pathlib import Path
from typing import Dict, Optional, NamedTuple
from PIL import Image


class Item(NamedTuple):
    image: Image.Image
    idx: Optional[int] = None
    name: Optional[str] = None


class TrainSplit(data.Dataset):
    def __init__(self, root: Path, class_map):
        super().__init__()
        self.root = root
        self._class_map = class_map
        self.image_paths = [*self.root.glob("**/*.JPEG")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        class_id = path.name.split("_")[0]
        class_idx, class_name = self._class_map[class_id]
        image = Image.open(path)
        return Item(image=image, idx=class_idx, name=class_name)


class ValSplit(data.Dataset):
    def __init__(self, root: Path, class_map):
        super().__init__()
        self.root = root
        self._class_map = class_map
        self.image_paths = [*(self.root / "images").iterdir()]
        self._annotations = self._parse_annotations()

    def _parse_annotations(self):
        _annotations = {}
        with open(self.root / "val_annotations.txt", "r") as f:
            for line in f:
                words = line.strip().split()
                name, class_id = words[:2]
                _annotations[name] = self._class_map[class_id]
        return _annotations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        class_idx, class_name = self._annotations[path.name]
        image = Image.open(path)
        return Item(image=image, idx=class_idx, name=class_name)


class TestSplit(data.Dataset):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.image_paths = [*(self.root / "images").iterdir()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        return Item(image=image)


class TinyImageNet(data.Dataset):
    def __init__(self, root: os.PathLike, split: str):
        super().__init__()
        self.root = Path(root)
        if split == "train":
            class_map = self._fetch_class_map()
            self._ds = TrainSplit(self.root / "train", class_map)
        elif split == "val":
            class_map = self._fetch_class_map()
            self._ds = ValSplit(self.root / "val", class_map)
        elif split == "test":
            self._ds = TestSplit(self.root / "test")

    def _fetch_class_map(self):
        class_map = {}
        with open(self.root / "words.txt", "r") as words_f:
            for line in words_f:
                words = line.strip().split()
                class_id = words[0]
                class_name = " ".join(words[1:])
                class_map[class_id] = class_name

        with open(self.root / "wnids.txt", "r") as wnids_f:
            classes = [line.strip() for line in wnids_f]

        class_map = {id: (idx, class_map[id]) for idx, id in enumerate(classes)}
        return class_map

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]
