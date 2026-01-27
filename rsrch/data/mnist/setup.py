from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download


class Args(BaseModel):
    source: str = "http://yann.lecun.com/exdb/mnist/"
    """Source URL for the dataset."""
    data_root: str
    """Output directory in which to place the dataset."""


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    for name in (
        "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ):
        source = urljoin(args.source, name)
        download(
            url=source,
            dest=data_root / name,
        )


if __name__ == "__main__":
    main()
