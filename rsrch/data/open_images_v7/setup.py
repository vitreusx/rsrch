from pathlib import Path, PurePosixPath

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download, download_and_extract


class Args(BaseModel):
    fetch_masks: bool = True
    """Whether to fetch segmentation masks."""
    data_root: str
    """Output directory in which to place the dataset."""


IMAGE_IDS = {
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "val": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}


IMAGE_LABELS = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv",
    "val": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels-boxable.csv",
}


BOXES = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "val": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}


SEG_META = {
    "train": "https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv",
    "val": "https://storage.googleapis.com/openimages/v5/validation-annotations-object-segmentation.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-object-segmentation.csv",
}

SEG_MASK_ARCHIVES = {
    "train": [
        f"https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{shard}.zip"
        for shard in "0123456789abcdef"
    ],
    "val": [
        f"https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-{shard}.zip"
        for shard in "0123456789abcdef"
    ],
    "test": [
        f"https://storage.googleapis.com/openimages/v5/test-masks/test-masks-{shard}.zip"
        for shard in "0123456789abcdef"
    ],
}

METADATA = [
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "https://storage.googleapis.com/openimages/v7/oidv7-classes-segmentation.txt",
]


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    # Get general metadata
    for url in METADATA:
        dest = data_root / "metadata" / PurePosixPath(url).name
        download(url, dest)

    # Get image IDs
    for split, url in IMAGE_IDS.items():
        dest = data_root / split / "labels" / "image_ids.csv"
        download(url, dest)

    # Get image labels
    for split, url in IMAGE_LABELS.items():
        dest = data_root / split / "labels" / "image_labels.csv"
        download(url, dest)

    # Get bboxes
    for split, url in BOXES.items():
        dest = data_root / split / "labels" / "boxes.csv"
        download(url, dest)

    # Get segmentation metadata
    for split, url in SEG_META.items():
        dest = data_root / split / "labels" / "segmentation.csv"
        download(url, dest)

    if args.fetch_masks:
        for split, urls in SEG_MASK_ARCHIVES.items():
            dest = data_root / split / "masks"
            for url in urls:
                download_and_extract(url, dest)


if __name__ == "__main__":
    main()
