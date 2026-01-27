import shutil
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download_and_extract


class Args(BaseModel):
    source: str = "http://images.cocodataset.org"
    """Source URL from which to fetch the archives."""
    modules: tuple[str, ...] = ("detection", "panoptic", "stuff")
    """COCO sub-datasets to fetch."""
    remove_archives: bool = False
    """Whether to remove the downloaded archives after extraction."""
    data_root: str
    """Output directory in which to place the dataset."""


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)
    ann_dir = data_root / "annotations"
    archive_dest = None if args.remove_archives else data_root

    for name in ("train2017", "val2017"):
        source = urljoin(args.source, f"zips/{name}.zip")
        download_and_extract(
            url=source,
            dest=data_root,
            archive_dest=archive_dest,
        )

    if "detections" in args.modules:
        source = urljoin(args.source, "annotations/annotations_trainval2017.zip")
        download_and_extract(
            url=source,
            dest=data_root,
            archive_dest=archive_dest,
        )

    if "panoptic" in args.modules:
        source = urljoin(
            args.source, "annotations/panoptic_annotations_trainval2017.zip"
        )
        download_and_extract(
            url=source,
            dest=data_root,
            archive_dest=archive_dest,
        )

        shutil.rmtree(data_root / "__MACOSX")
        shutil.rmtree(ann_dir / ".DS_Store")
        for split in ("train", "val"):
            split_zip = ann_dir / f"panoptic_{split}2017.zip"
            with zipfile.ZipFile(split_zip) as sf:
                sf.extractall(ann_dir)
            split_zip.unlink()

    if "stuff" in args.modules:
        source = urljoin(args.source, "annotations/stuff_annotations_trainval2017.zip")
        download_and_extract(
            url=source,
            dest=data_root,
            archive_dest=archive_dest,
        )

        shutil.rmtree(ann_dir / "deprecated-challenge")
        for split in ("train", "val"):
            split_zip = ann_dir / f"stuff_{split}2017_pixelmaps.zip"
            with zipfile.ZipFile(split_zip) as sf:
                sf.extractall(ann_dir)
            split_zip.unlink()


if __name__ == "__main__":
    main()
